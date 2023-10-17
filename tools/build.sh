#!/bin/bash

export PATH=/opt/python/cp37-cp37m/bin:$PATH
export BUILD_TYPE=Release

pip install conan==1.58 ninja
conan install . -if build --build=missing -s build_type=$BUILD_TYPE --profile=default -o runtime=False -o python=False -o tests=True
conan user sunnycase -r sunnycase -p $1
conan upload "*" --all -r sunnycase -c
