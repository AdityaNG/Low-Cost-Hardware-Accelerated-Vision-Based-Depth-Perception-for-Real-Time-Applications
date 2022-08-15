#import setuptools
from distutils.core import setup, Extension
import subprocess
import glob
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


UNAME = subprocess.run(['uname', '-m'], stdout=subprocess.PIPE).stdout.decode('utf-8').replace('\n', '')

OPENCV_INCLUDE_PATH = list(map(lambda i: i.replace('-I', '') ,subprocess.run(['pkg-config', '--cflags', 'opencv'], stdout=subprocess.PIPE).stdout.decode('utf-8').replace('\n', '').split()))
OPENCV_LINKER_ARGS = subprocess.run(['pkg-config', '--libs', 'opencv'], stdout=subprocess.PIPE).stdout.decode('utf-8').replace('\n', '').split()
# OPENCV_INCLUDE_PATH = list(map(lambda i: i.replace('-I', '') ,subprocess.run(['pkg-config', '--cflags', 'opencv4'], stdout=subprocess.PIPE).stdout.decode('utf-8').replace('\n', '').split()))
# OPENCV_LINKER_ARGS = subprocess.run(['pkg-config', '--libs', 'opencv4'], stdout=subprocess.PIPE).stdout.decode('utf-8').replace('\n', '').split()

if 'armv7l' in UNAME: # Raspberry Pi
    OPENCV_INCLUDE_PATH = list(map(lambda i: i.replace('-I', '') ,subprocess.run(['pkg-config', '--cflags', 'opencv4'], stdout=subprocess.PIPE).stdout.decode('utf-8').replace('\n', '').split()))
    OPENCV_LINKER_ARGS = subprocess.run(['pkg-config', '--libs', 'opencv4'], stdout=subprocess.PIPE).stdout.decode('utf-8').replace('\n', '').split()

stereo_vision_serial_module = Extension('stereo_vision_serial',
                    include_dirs = ['/usr/local/include'] + OPENCV_INCLUDE_PATH,
                    library_dirs = ['/usr/local/lib'],
                    extra_link_args= ['-lpopt', '-lglut', '-lGLU', '-lGL', '-lm', '-lpthread', '-fopenmp' ] + OPENCV_LINKER_ARGS,
                    extra_compile_args=['-O3', '-std=c++17', '-w'],
                    sources = glob.glob("src/common_includes/*/*.cpp") + glob.glob("src/serial_includes/*/*.cpp")
                    )

setup(
    name="stereo_vision",
    version="0.0.1",
    author="Aditya NG, Dhruval PB",
    author_email="adityang5@gmail.com, dhruvalpb@gmail.com",
    packages=['stereo_vision'],
    package_dir={'stereo_vision': 'stereo_vision'},
    include_package_data=True,
    # package_data={'stereo_vision': glob.glob("bin/*") + glob.glob("data/*")},
	package_data={'stereo_vision': glob.glob("data/*")},
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
