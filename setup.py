#import setuptools
from distutils.core import setup, Extension
import subprocess
import glob
from os import path
import os
from os.path import join as pjoin
import numpy

def find_in_path(name, path):
    "Find a file in a search path"
    #adapted fom http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


def locate_cuda():
    """Locate the CUDA environment on the system
    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.
    Starts by looking for the CUDAHOME env variable. If not found, everything
    is based on finding 'nvcc' in the PATH.
    """

    # first check if the CUDAHOME env variable is in use
    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc = pjoin(home, 'bin', 'nvcc')
    else:
        # otherwise, search the PATH for NVCC
        nvcc = find_in_path('nvcc', os.environ['PATH'])
        if nvcc is None:
            raise EnvironmentError('The nvcc binary could not be '
                'located in your $PATH. Either add it to your path, or set $CUDAHOME')
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {'home':home, 'nvcc':nvcc,
                  'include': pjoin(home, 'include'),
                  'lib64': pjoin(home, 'lib64')}
    for k, v in cudaconfig.items():
        if not os.path.exists(v):
            raise EnvironmentError('The CUDA %s path could not be located in %s' % (k, v))

    return cudaconfig
CUDA = locate_cuda()


# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()


this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

OPENCV_INCLUDE_PATH = list(map(lambda i: i.replace('-I', '') ,subprocess.run(['pkg-config', '--cflags', 'opencv'], stdout=subprocess.PIPE).stdout.decode('utf-8').replace('\n', '').split()))
OPENCV_LINKER_ARGS = subprocess.run(['pkg-config', '--libs', 'opencv'], stdout=subprocess.PIPE).stdout.decode('utf-8').replace('\n', '').split()
stereo_vision_serial_module = Extension('stereo_vision_serial',
                    include_dirs = ['/usr/local/include'] + OPENCV_INCLUDE_PATH,
                    library_dirs = ['/usr/local/lib'],
                    extra_link_args= ['-lpopt', '-lglut', '-lGLU', '-lGL', '-lm', '-lpthread', '-fopenmp' ] + OPENCV_LINKER_ARGS,
                    extra_compile_args=['-O3', '-std=c++17', '-w'],
                    sources = glob.glob("src/common_includes/*/*.cpp") + glob.glob("src/serial_includes/*/*.cpp")
                    )

stereo_vision_parallel_module = Extension('stereo_vision_serial',
                    
                    library_dirs = [CUDA['lib64']],
                    libraries = ['cudart'],
                    runtime_library_dirs = [CUDA['lib64']],
                    extra_compile_args={'gcc': ['-O3', '-std=c++17', '-w'],
                                        'nvcc': ['-O3', '-std=c++17', '-w'] + ['--ptxas-options=-v', '-c', '--compiler-options', "'-fPIC'"]},
                    include_dirs = [numpy_include, CUDA['include'], 'src'] + OPENCV_INCLUDE_PATH,
                    extra_link_args= ['-lpopt', '-lglut', '-lGLU', '-lGL', '-lm', '-lpthread', '-fopenmp' ] + OPENCV_LINKER_ARGS,
                    sources = glob.glob("src/common_includes/*/*.cpp") + glob.glob("src/parallel_includes/*/*.cpp") + glob.glob("src/parallel_includes/*/*.cu")
                    )

if find_in_path('swig', os.environ['PATH']):
    subprocess.check_call('swig -python -c++ -o src/common_includes/bayesian/bayesian.cpp', shell=True)
else:
    raise EnvironmentError('the swig executable was not found in your PATH')

setup(
    name="stereo_vision",
    version="0.0.1",
    author="Aditya NG, Dhruval PB",
    author_email="adityang5@gmail.com, dhruvalpb@gmail.com",
    packages=['stereo_vision'],
    package_dir={'stereo_vision': 'stereo_vision'},
    include_package_data=True,
    package_data={'stereo_vision': glob.glob("bin/*") + glob.glob("data/*")},
    #scripts=['bin/vivp'],
    license='LICENSE.txt',
    url='https://github.com/AdityaNG/Depth-Perception-from-Stereoscopic-Vision-on-Edge-Devices',
    description="A library to simplify disparity calculation and 3D depth map generation from a stereo pair",
    long_description=long_description,
    long_description_content_type='text/markdown',
    python_requires='>=3.6',
    ext_modules = [stereo_vision_serial_module]
)

"""
To upload :
rm -rf dist/
python3 setup.py sdist bdist_wheel
python3 -m twine upload --repository-url https://upload.pypi.org/legacy/ dist/*
"""
