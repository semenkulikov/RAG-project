"""
Setup script for RAG Legal Document Generator
"""

from setuptools import setup, find_packages
from pathlib import Path

# Читаем README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Читаем requirements
requirements = []
with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="rag-legal-generator",
    version="1.0.0",
    author="RAG System Team",
    author_email="team@rag-system.com",
    description="Система генерации юридических документов с использованием RAG",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rag-system/legal-generator",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Legal Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Legal",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "gpu": [
            "torch>=1.9.0",
            "torchvision>=0.10.0",
            "torchaudio>=0.9.0",
        ],
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
    },
    entry_points={
        "console_scripts": [
            "rag-generator=src.rag_system.legal_document_generator:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
)
