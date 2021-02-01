from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)

import os
import setuptools
import versioneer

here = os.path.abspath(os.path.dirname(__file__))

# Dependencies.
with open("requirements.txt") as f:
    requirements = f.readlines()
install_requires = [t.strip() for t in requirements]

with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name="ooipy",
    version=versioneer.get_version(),
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
    install_requires=install_requires,
    cmdclass=versioneer.get_cmdclass()
)