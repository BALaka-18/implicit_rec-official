import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="implicit_reca", 
    version="0.0.1",
    author="Balaka Biswas",
    author_email="balaka2605@gmail.com",
    description="Package for implicit recommendations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BALaka-18/implicit_rec-official",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)