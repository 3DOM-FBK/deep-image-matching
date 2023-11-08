from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="custom_ops",
    ext_modules=[
        CUDAExtension("get_patches", ["get_patches.cpp", "get_patches_cuda.cu"])
    ],
    cmdclass={"build_ext": BuildExtension},
)
