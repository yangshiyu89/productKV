import math
import numpy as np
import torch
import safetensors.torch as sf

from PIL import Image, ImageFilter
from diffusers import StableDiffusionPipeline
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer

import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.dirname(__dir__))
from models import BriaRMBG


@torch.inference_mode()
def pytorch2numpy(imgs, quant=True):
    results = []
    for x in imgs:
        y = x.movedim(0, -1)

        if quant:
            y = y * 127.5 + 127.5
            y = y.detach().float().cpu().numpy().clip(0, 255).astype(np.uint8)
        else:
            y = y * 0.5 + 0.5
            y = y.detach().float().cpu().numpy().clip(0, 1).astype(np.float32)

        results.append(y)
    return results


@torch.inference_mode()
def numpy2pytorch(imgs):
    h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.0 - 1.0  # so that 127 must be strictly 0.0
    h = h.movedim(-1, 1)
    return h


def resize_and_center_crop(image, target_width, target_height):
    pil_image = Image.fromarray(image)
    original_width, original_height = pil_image.size
    scale_factor = max(target_width / original_width, target_height / original_height)
    resized_width = int(round(original_width * scale_factor))
    resized_height = int(round(original_height * scale_factor))
    resized_image = pil_image.resize((resized_width, resized_height), Image.LANCZOS)
    left = (resized_width - target_width) / 2
    top = (resized_height - target_height) / 2
    right = (resized_width + target_width) / 2
    bottom = (resized_height + target_height) / 2
    cropped_image = resized_image.crop((left, top, right, bottom))
    return np.array(cropped_image)


def resize_without_crop(image, target_width, target_height):
    pil_image = Image.fromarray(image)
    resized_image = pil_image.resize((target_width, target_height), Image.LANCZOS)
    return np.array(resized_image)

        
class ProductKV:
    
    def __init__(self, device="cuda:0"):
        self.device = device
        
    def load_models(
        self,
        sd15_base_path,
        rmbg_path,
        ic_light_path
    ):
        self.tokenizer = CLIPTokenizer.from_pretrained(sd15_base_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(sd15_base_path, subfolder="text_encoder")
        self.vae = AutoencoderKL.from_pretrained(sd15_base_path, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(sd15_base_path, subfolder="unet")
        
        self.rmbg = BriaRMBG.from_pretrained(rmbg_path)
        
        # Change UNet

        with torch.no_grad():
            new_conv_in = torch.nn.Conv2d(8, self.unet.conv_in.out_channels, self.unet.conv_in.kernel_size, self.unet.conv_in.stride, self.unet.conv_in.padding)
            new_conv_in.weight.zero_()
            new_conv_in.weight[:, :4, :, :].copy_(self.unet.conv_in.weight)
            new_conv_in.bias = self.unet.conv_in.bias
            self.unet.conv_in = new_conv_in
            
        unet_original_forward = self.unet.forward
        
        def hooked_unet_forward(sample, timestep, encoder_hidden_states, **kwargs):
            c_concat = kwargs["cross_attention_kwargs"]["concat_conds"].to(sample)
            c_concat = torch.cat([c_concat] * (sample.shape[0] // c_concat.shape[0]), dim=0)
            new_sample = torch.cat([sample, c_concat], dim=1)
            kwargs["cross_attention_kwargs"] = {}
            return unet_original_forward(new_sample, timestep, encoder_hidden_states, **kwargs)
        
        self.unet.forward = hooked_unet_forward
        
        sd_offset = sf.load_file(ic_light_path)
        sd_origin = self.unet.state_dict()
        keys = sd_origin.keys()
        sd_merged = {k: sd_origin[k] + sd_offset[k] for k in sd_origin.keys()}
        self.unet.load_state_dict(sd_merged, strict=True)
        del sd_offset, sd_origin, sd_merged, keys
        
        device = self.device
        self.text_encoder = self.text_encoder.to(device=device, dtype=torch.float16)
        self.vae = self.vae.to(device=device, dtype=torch.bfloat16)
        self.unet = self.unet.to(device=device, dtype=torch.float16)
        self.rmbg = self.rmbg.to(device=device, dtype=torch.float32)
        
        self.unet.set_attn_processor(AttnProcessor2_0())
        self.vae.set_attn_processor(AttnProcessor2_0())
        
        
        # Samplers

        self.ddim_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )

        self.euler_a_scheduler = EulerAncestralDiscreteScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            steps_offset=1
        )

        self.dpmpp_2m_sde_karras_scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            algorithm_type="sde-dpmsolver++",
            use_karras_sigmas=True,
            steps_offset=1
        )
        
        # Pipelines

        self.t2i_pipe = StableDiffusionPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            scheduler=self.dpmpp_2m_sde_karras_scheduler,
            safety_checker=None,
            requires_safety_checker=False,
            feature_extractor=None,
            image_encoder=None
        )
    
    @torch.inference_mode()
    def encode_prompt_inner(self, txt: str):
        max_length = self.tokenizer.model_max_length
        chunk_length = self.tokenizer.model_max_length - 2
        id_start = self.tokenizer.bos_token_id
        id_end = self.tokenizer.eos_token_id
        id_pad = id_end

        def pad(x, p, i):
            return x[:i] if len(x) >= i else x + [p] * (i - len(x))

        tokens = self.tokenizer(txt, truncation=False, add_special_tokens=False)["input_ids"]
        chunks = [[id_start] + tokens[i: i + chunk_length] + [id_end] for i in range(0, len(tokens), chunk_length)]
        chunks = [pad(ck, id_pad, max_length) for ck in chunks]

        token_ids = torch.tensor(chunks).to(device=self.device, dtype=torch.int64)
        conds = self.text_encoder(token_ids).last_hidden_state

        return conds
    
    @torch.inference_mode()
    def encode_prompt_pair(self, positive_prompt, negative_prompt):
        c = self.encode_prompt_inner(positive_prompt)
        uc = self.encode_prompt_inner(negative_prompt)

        c_len = float(len(c))
        uc_len = float(len(uc))
        max_count = max(c_len, uc_len)
        c_repeat = int(math.ceil(max_count / c_len))
        uc_repeat = int(math.ceil(max_count / uc_len))
        max_chunk = max(len(c), len(uc))

        c = torch.cat([c] * c_repeat, dim=0)[:max_chunk]
        uc = torch.cat([uc] * uc_repeat, dim=0)[:max_chunk]

        c = torch.cat([p[None, ...] for p in c], dim=1)
        uc = torch.cat([p[None, ...] for p in uc], dim=1)

        return c, uc


    @torch.inference_mode()
    def run_rmbg(self, img_pil, sigma=0.0):
        img = np.asarray(img_pil)
        H, W, C = img.shape
        assert C == 3
        k = (256.0 / float(H * W)) ** 0.5
        feed = resize_without_crop(img, int(64 * round(W * k)), int(64 * round(H * k)))
        feed = numpy2pytorch([feed]).to(device=self.device, dtype=torch.float32)
        alpha = self.rmbg(feed)[0][0]
        alpha = torch.nn.functional.interpolate(alpha, size=(H, W), mode="bilinear")
        alpha = alpha.movedim(1, -1)[0]
        alpha = alpha.detach().float().cpu().numpy().clip(0, 1)
        result = 127 + (img.astype(np.float32) - 127 + sigma) * alpha
        result = result.clip(0, 255).astype(np.uint8)
        mask = (alpha * 255).astype(np.uint8)
        mask = np.tile(mask, (1, 1, 3))
        mask = Image.fromarray(mask)
        return Image.fromarray(result), mask


    @torch.inference_mode()
    def __call__(
        self, 
        image, 
        product_prompt, 
        background_prompt,
        image_width = 1024, 
        image_height = 1024, 
        num_samples = 1, 
        seed = -1, 
        steps = 25, 
        a_prompt = "best quality", 
        n_prompt = "lowres, bad anatomy, bad hands, cropped, worst quality",
        cfg = 2
    ):
        
        prompt = product_prompt + ", " + background_prompt + " background"
        
        input_fg, mask = self.run_rmbg(image)

        if seed <= 0:
            seed = np.random.randint(0, 2**31 - 1)
        rng = torch.Generator(device=self.device).manual_seed(int(seed))

        fg = resize_and_center_crop(np.asarray(input_fg), image_width, image_height)

        concat_conds = numpy2pytorch([fg]).to(device=self.vae.device, dtype=self.vae.dtype)
        concat_conds = self.vae.encode(concat_conds).latent_dist.mode() * self.vae.config.scaling_factor

        conds, unconds = self.encode_prompt_pair(positive_prompt=prompt + ", " + a_prompt, negative_prompt=n_prompt)

        latents = self.t2i_pipe(
            prompt_embeds=conds,
            negative_prompt_embeds=unconds,
            width=image_width,
            height=image_height,
            num_inference_steps=steps,
            num_images_per_prompt=num_samples,
            generator=rng,
            output_type="latent",
            guidance_scale=cfg,
            cross_attention_kwargs={"concat_conds": concat_conds},
        ).images.to(self.vae.dtype) / self.vae.config.scaling_factor


        pixels = self.vae.decode(latents).sample


        generated_result = Image.fromarray(pytorch2numpy(pixels)[0])
        
        mask = mask.convert("L")
        mask_small = mask.filter(ImageFilter.MinFilter(size = 51))
        mask_blur = mask_small.filter(ImageFilter.GaussianBlur(radius = 25))
        cutout = Image.composite(image, generated_result, mask_blur)
        blend_result = Image.blend(cutout, generated_result, 0.4)
        
        return blend_result
    
