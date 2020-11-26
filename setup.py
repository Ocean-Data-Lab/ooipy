import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ooipy",
    version="1.0.2",
    author="OOIPy",
    author_email="ooipython@gmail.com",
    description="A python toolbox for acquiring and analyzing Ocean Obvservatories Initiative (OOI) Data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ooipy/ooipy",
    packages=setuptools.find_packages(exclude=("tests")),
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)