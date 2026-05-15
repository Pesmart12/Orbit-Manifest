import sys
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

if sys.platform == "win32":
    extra_compile_args = ["/O2", "/fp:fast", "/openmp"]
    extra_link_args = []
else:
    extra_compile_args = ["-O3", "-ffast-math", "-fopenmp"]
    extra_link_args = ["-fopenmp"]

ext = Pybind11Extension(
    "orbit_integrator",
    sources=[
        "integrator/integrator.cpp",
        "integrator/bindings.cpp",
    ],
    include_dirs=["integrator"],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    cxx_std=17,
)

setup(
    name="orbit-manifest",
    version="0.1.0",
    ext_modules=[ext],
    cmdclass={"build_ext": build_ext},
)
