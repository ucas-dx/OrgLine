from setuptools import setup, find_packages
import os

# Safely read README.md
def get_long_description():
    if os.path.exists("README.md"):
        try:
            with open("README.md", "r", encoding="utf-8") as fh:
                return fh.read()
        except Exception:
            pass
    return "OrgLine, the first multitask analysis pipeline for organoids, which leverages a pretrained detector and a vision foundational model to facilitate large-scale automated cultivation and analysis of organoids."

# Read requirements.txt and parse dependencies
def get_requirements():
    requirements = []
    if os.path.exists("requirements.txt"):
        try:
            with open("requirements.txt", "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    # Skip empty lines, comments, and special directives
                    if line and not line.startswith("#") and not line.startswith("-"):
                        # Handle special PyTorch install instructions
                        if "torch" in line and "--index-url" not in line:
                            requirements.append(line)
                        elif "torch" not in line and "git+" not in line:
                            requirements.append(line)
        except Exception as e:
            print(f"Warning: Could not read requirements.txt: {e}")
    
    # Provide basic dependencies if requirements.txt is missing
    if not requirements:
        requirements = [
            "numpy>=1.21.0",
            "opencv-python>=4.5.0",
            "pillow>=8.0.0",
            "matplotlib>=3.3.0",
            "pandas>=1.3.0",
            "tqdm>=4.62.0",
            "scikit-learn>=1.0.0",
        ]
    
    return requirements

setup(
    name="orgline",
    version="0.1.0",
    author="Xun Deng, Xinyu Hao, Pengwei Hu", 
    author_email="hpw@ms.xjb.ac.cn",
    description="The first multitask analysis pipeline for organoids with foundation models",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/ucas-dx/OrgLine",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    keywords="organoid, analysis, detection, segmentation, foundation model, computer vision, biomedical imaging",
    python_requires=">=3.8",
    
    # Automatically install dependencies from requirements.txt
    install_requires=get_requirements(),
    
    # Optional extra dependencies
    extras_require={
        "torch": [
            "torch>=1.12.0",
            "torchvision>=0.13.0",
            "torchaudio>=0.12.0",
        ],
        "cuda118": [
            "torch==2.3.0",
            "torchvision==0.18.0", 
            "torchaudio==2.3.0",
        ],
        "full": [
            "torch>=1.12.0",
            "torchvision>=0.13.0",
            "einops>=0.6.0",
            "timm>=0.9.0",
            "albumentations>=1.0.0",
            "wandb>=0.12.0",
            "jupyter>=1.0.0",
            "notebook>=6.0.0",
        ],
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.910",
            "pre-commit>=2.15.0",
        ],
    },
    
    # CLI tool entry points
    entry_points={
        "console_scripts": [
            "orgline-analyze=models.simple_inference:main",
        ],
    },
    
    # Include data files
    include_package_data=True,
    package_data={
        "": [
            "*.yaml",
            "*.yml", 
            "*.json",
            "*.md",
            "images/*",
            "models/*.py",
        ],
    },
    
    # Project URLs
    project_urls={
        "Homepage": "https://github.com/ucas-dx/OrgLine",
        "Source Code": "https://github.com/ucas-dx/OrgLine",
        "Bug Reports": "https://github.com/ucas-dx/OrgLine/issues",
        "Documentation": "https://github.com/ucas-dx/OrgLine#readme",
    },
    
    zip_safe=False,
)