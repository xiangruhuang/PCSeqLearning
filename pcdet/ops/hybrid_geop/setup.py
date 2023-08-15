from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

cuda_module = CUDAExtension(
                  'hybrid_geop_cuda',
                  sources = [
                      'src/svd3_kernel.cu',
                      'src/hybrid_geop_api.cpp',
                      'src/hybrid_geop_kernel.cu',
                  ],
                  extra_compile_args={
                      'cxx': ['-g', '-I /usr/local/cuda/include'],
                      'nvcc': ['-O2'],
                  },
              )

setup(
    name='hybrid_geop',
    ext_modules=[cuda_module],
    cmdclass={'build_ext': BuildExtension}
)
