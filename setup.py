from setuptools import find_packages, setup

def fetch_requirements():
    with open("requirements.txt") as f:
        reqs = f.read().strip().split("\n")
    return reqs
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="productKV",
    version="0.0.1",
    author="yangshiyu89",
    author_email="yangshiyu89@sina.com",
    description="product kv cover package",
    long_description="an AI tool for product KV cover image",
    long_description_content_type="text/markdown",
    package_dir = {"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.9.0",
    url="",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: OS Independent"
        ],
    )