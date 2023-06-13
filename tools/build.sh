#!/bin/bash

export PATH=/opt/python/cp37-cp37m/bin:$PATH
export BUILD_TYPE=Release

pip install conan==1.58 ninja
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=$BUILD_TYPE -BUILD_PYTHON_BINDING=OFF
cmake --build build --config $BUILD_TYPE
cmake --install build --prefix install
conan user sunnycase -r sunnycase -p $1
conan upload "*" --all -r sunnycase -c
