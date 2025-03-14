from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="static-dynamic-bert-autopytorch",
    version="0.1.0",
    author="Parisa Safikhani",
    author_email="parisa.safikhani@ovgu.de",
    description="Static and dynamic contextual embedding for AutoML in text classification tasks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ParisaSafikhani/StaticDynamic-BERT-AutoPyTorch",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=requirements,
) 