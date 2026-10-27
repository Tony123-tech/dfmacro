import numpy as np
from setuptools import Extension, setup

module = Extension(
    "dfmacro_c", sources=["dfmacro_c.c"], include_dirs=[np.get_include()]
)

setup(name="dfmacro_c", version="1.0", ext_modules=[module])
