import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="finetwork",
    version="1.0.0",
    author="annakuchko",
    author_email="anna (dot) kuchko at yandex (dot) ru",
    description="Portfolio optimisation based on financial network analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/annakuchko/FiNetwork",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
