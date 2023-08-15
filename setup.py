import os
import subprocess

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def get_git_commit_number():
    if not os.path.exists('.git'):
        return '0000000'

    cmd_out = subprocess.run(['git', 'rev-parse', 'HEAD'], stdout=subprocess.PIPE)
    git_commit_number = cmd_out.stdout.decode('utf-8')[:7]
    return git_commit_number


def make_cuda_ext(name, module, sources, **kwargs):
    cuda_ext = CUDAExtension(
        name='%s.%s' % (module, name),
        sources=[os.path.join(*module.split('.'), src) for src in sources],
        **kwargs
    )
    return cuda_ext


def write_version_to_file(version, target_file):
    with open(target_file, 'w') as f:
        print('__version__ = "%s"' % version, file=f)


if __name__ == '__main__':
    version = '0.5.2+%s' % get_git_commit_number()
    write_version_to_file(version, 'pcdet/version.py')

    setup(
        name='pcdet',
        version=version,
        description='OpenPCDet is a general codebase for 3D object detection from point cloud',
        install_requires=[
            'numpy',
            'llvmlite',
            'numba',
            'tensorboardX',
            'easydict',
            'pyyaml',
            'scikit-image',
            'tqdm',
            'SharedArray',
            # 'spconv',  # spconv has different names depending on the cuda version
        ],

        author='Shaoshuai Shi',
        author_email='shaoshuaics@gmail.com',
        license='Apache License 2.0',
        packages=find_packages(exclude=['tools', 'data', 'output']),
        cmdclass={
            'build_ext': BuildExtension,
        },
        ext_modules=[
            make_cuda_ext(
                name='sparse_conv_ext',
                module='pcdet.ops.spconv',
                sources=[
                    'src/all.cc',
                    'src/reordering.cc',
                    'src/reordering_cuda.cu',
                    'src/indice.cc',
                    'src/indice_cuda.cu',
                    'src/maxpool.cc',
                    'src/maxpool_cuda.cu',
                ],
                extra_compile_args=['-w', '-std=c++14'],
                include_dirs=[
                    os.path.abspath('pcdet/ops/spconv/include/')
                ],
                ),
            make_cuda_ext(
                name='iou3d_nms_cuda',
                module='pcdet.ops.iou3d_nms',
                sources=[
                    'src/iou3d_cpu.cpp',
                    'src/iou3d_nms_api.cpp',
                    'src/iou3d_nms.cpp',
                    'src/iou3d_nms_kernel.cu',
                ]
            ),
            make_cuda_ext(
                name='roiaware_pool3d_cuda',
                module='pcdet.ops.roiaware_pool3d',
                sources=[
                    'src/roiaware_pool3d.cpp',
                    'src/roiaware_pool3d_kernel.cu',
                ]
            ),
            make_cuda_ext(
                name='roipoint_pool3d_cuda',
                module='pcdet.ops.roipoint_pool3d',
                sources=[
                    'src/roipoint_pool3d.cpp',
                    'src/roipoint_pool3d_kernel.cu',
                ]
            ),
            make_cuda_ext(
                name='pointnet2_stack_cuda',
                module='pcdet.ops.pointnet2.pointnet2_stack',
                sources=[
                    'src/pointnet2_api.cpp',
                    'src/ball_query.cpp',
                    'src/ball_query_gpu.cu',
                    'src/group_points.cpp',
                    'src/group_points_gpu.cu',
                    'src/sampling.cpp',
                    'src/sampling_gpu.cu', 
                    'src/interpolate.cpp', 
                    'src/interpolate_gpu.cu',
                    'src/voxel_query.cpp', 
                    'src/voxel_query_gpu.cu',
                    'src/vector_pool.cpp',
                    'src/vector_pool_gpu.cu'
                ],
            ),
            make_cuda_ext(
                name='pointnet2_batch_cuda',
                module='pcdet.ops.pointnet2.pointnet2_batch',
                sources=[
                    'src/pointnet2_api.cpp',
                    'src/ball_query.cpp',
                    'src/ball_query_gpu.cu',
                    'src/group_points.cpp',
                    'src/group_points_gpu.cu',
                    'src/interpolate.cpp',
                    'src/interpolate_gpu.cu',
                    'src/sampling.cpp',
                    'src/sampling_gpu.cu',
                ],
            ),
            make_cuda_ext(
                name='torch_hash_cuda',
                module='pcdet.ops.torch_hash',
                sources=[
                    'src/torch_hash_api.cpp',
                    'src/torch_hash_kernel.cu',
                ],
            ),
            make_cuda_ext(
                name='virtual_array_cuda',
                module='pcdet.ops.virtual_array',
                sources=[
                    'src/virtual_array_api.cpp',
                    'src/virtual_array_kernel.cu',
                ],
            ),
            make_cuda_ext(
                name='sparse_kpconv_cuda',
                module='pcdet.ops.sparse_kpconv',
                sources=[
                    'src/sparse_kpconv_api.cpp',
                    'src/sparse_kpconv_kernel.cu',
                ],
            ),
            make_cuda_ext(
                name='hybrid_geop_cuda',
                module='pcdet.ops.hybrid_geop',
                sources = [
                    'src/svd3_kernel.cu',
                    'src/hybrid_geop_api.cpp',
                    'src/hybrid_geop_kernel.cu',
                ],
                extra_compile_args={
                    'cxx': ['-g', '-I /usr/local/cuda/include'],
                    'nvcc': ['-O2'],
                },
            ),
        ],
    )
