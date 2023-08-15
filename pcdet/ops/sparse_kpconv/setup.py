from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

cuda_module = CUDAExtension(
                  'sparse_kpconv_cuda',
                  sources = [
                      'src/sparse_kpconv_api.cpp',
                      'src/sparse_kpconv_kernel.cu',
                  ],
                  extra_compile_args={
                      'cxx': ['-g', '-I /usr/local/cuda/include'],
                      'nvcc': ['-O2'],
                  },
              )

setup(
    name='sparse_kpconv',
    ext_modules=[cuda_module],
    cmdclass={'build_ext': BuildExtension}
)
