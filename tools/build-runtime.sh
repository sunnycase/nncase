#!/bin/bash

export PATH=/opt/python/cp37-cp37m/bin:$PATH
export BUILD_TYPE=Release

pip install conan==2.6.0
conan remote add sunnycase https://conan.sunnycase.moe --index 0
conan install . --build=missing -s build_type=$BUILD_TYPE -pr:a=toolchains/x86_64-linux.profile.jinja -o "&:runtime=True" -o "&:python=True" -o "&:tests=True"
conan remote login -p $1 sunnycase sunnycase
conan upload "*" -r sunnycase -c
