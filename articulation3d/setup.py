#!/usr/bin/env python3
from setuptools import find_packages, setup

setup(
    name="articulation3d",
    version="1.0",
    author="Shengyi Qian, Linyi Jin, Chris Rockwell, Siyi Chen, David Fouhey",
    description="Code for Understanding 3D Object Articulation in Internet Videos",
    packages=find_packages(exclude=("configs", "tests")),
    install_requires=["torchvision>=0.4", "fvcore", "detectron2", "pytorch3d"],
)