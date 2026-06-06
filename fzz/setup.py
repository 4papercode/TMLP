from setuptools import setup, Extension
import os.path as pth
import os



__module_file_dir = pth.dirname(pth.realpath(__file__))
__cpp_src_dir = pth.join(__module_file_dir, 'pyfzz')
src_files = []
src_files.append(pth.join(__cpp_src_dir, 'pyfzz.cpp'))
src_files.append(pth.join(__cpp_src_dir, '../fzz.cpp'))
setup(name='pyfzz',
      version='0.0.0',
      author='xx',
      description="xx",
      author_email='xx',
      python_requires='>=3.7',
      classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: BSD License',
        'Operating System :: MacOS :: MacOS X',
    ],
      ext_modules=[Extension('fzz',include_dirs=[os.path.join(__cpp_src_dir,'phat-include'), '..'],
                             sources=src_files, extra_compile_args=['-std=c++17'])],
                             )

