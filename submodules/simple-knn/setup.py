#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

# 使用了setuptools和torch.utils.cpp_extension来编译和链接C++和CUDA代码，并将其打包为一个Python模块
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

# 定义一个列表cxx_compiler_flags来存储 C++ 编译器标志
cxx_compiler_flags = []

if os.name == 'nt':
    cxx_compiler_flags.append("/wd4624")
# 调用setup函数来配置和安装Python包。
# 指定包的名称为simple_knn，指定扩展模块的名称为simple_knn._C，指定扩展模块的源文件为spatial.cu、simple_knn.cu和ext.cpp，指定扩展模块的编译标志为 cxx_compiler_flags。
# 通过cmdclass参数指定了一个字典，其中包含了一个键值对 'build_ext': BuildExtension，这个键值对指定了在构建扩展模块时使用BuildExtension 类。
setup(
    name="simple_knn",
    ext_modules=[
        CUDAExtension(
            name="simple_knn._C",
            sources=[
            "spatial.cu", 
            "simple_knn.cu",
            "ext.cpp"],
            extra_compile_args={"nvcc": [], "cxx": cxx_compiler_flags})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
