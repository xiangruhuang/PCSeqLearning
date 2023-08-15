from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

cuda_module = CUDAExtension(
                  'virtual_array_cuda',
                  sources = [
                      'src/virtual_array_api.cpp',
                      'src/virtual_array_kernel.cu',
                  ],
                  extra_compile_args={
                      'cxx': ['-g', '-I /usr/local/cuda/include'],
                      'nvcc': ['-O2'],
                  },
              )

setup(
    name='virtual_array',
    ext_modules=[cuda_module],
    cmdclass={'build_ext': BuildExtension}
)
