from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import sys
import os


class get_pybind_include:
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked."""

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)


ext_modules = [
    Extension(
        'hpda',
        ['src/python_bindings.cpp', 'src/dataframe.cpp'],
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True),
            "include",
        ],
        language='c++',
        extra_compile_args=['-std=c++17', '-O3'],
    ),
]


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': ['-O3', '-Wall', '-shared', '-std=c++17', '-fPIC'],
    }
    l_opts = {
        'msvc': [],
        'unix': [],
    }

    if sys.platform == 'darwin':
        darwin_opts = ['-stdlib=libc++']
        c_opts['unix'] += darwin_opts
        l_opts['unix'] += darwin_opts

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        link_opts = self.l_opts.get(ct, [])
        if ct == 'unix':
            opts.append(f'-DVERSION_INFO="{self.distribution.get_version()}"')
        elif ct == 'msvc':
            opts.append(f'/DVERSION_INFO=\\"{self.distribution.get_version()}\\"')
        for ext in self.extensions:
            ext.extra_compile_args = opts
            ext.extra_link_args = link_opts
        build_ext.build_extensions(self)


setup(
    name='hpda',
    version='0.1.0',
    description='High-performance data analysis library with pandas-like API',
    author='HP Data Analysis Team',
    author_email='info@hpda.com',
    url='https://github.com/yourusername/hp-data-analysis',
    packages=find_packages(),
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.6.0'],
    setup_requires=['pybind11>=2.6.0'],
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
    python_requires='>=3.7',
)