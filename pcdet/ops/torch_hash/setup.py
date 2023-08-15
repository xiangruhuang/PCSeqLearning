from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

cuda_module = CUDAExtension(
                  'torch_hash_cuda',
                  sources = [
                      'src/torch_hash_api.cpp',
                      'src/torch_hash_kernel.cu',
                  ],
                  extra_compile_args={
                      'cxx': ['-g', '-I /usr/local/cuda/include'],
                      'nvcc': ['-O2'],
                  },
              )

setup(
    name='torch_hash',
    ext_modules=[cuda_module],
    cmdclass={'build_ext': BuildExtension}
)
