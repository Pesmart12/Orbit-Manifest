cmake_minimum_required(VERSION 3.18)
project(orbit_integrator CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(MSVC)
  add_compile_options(/O2 /fp:fast)
else()
  add_compile_options(-O3 -ffast-math)
endif()

# Locate pybind11 — try cmake config first, fall back to python -m pybind11
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  find_package(pybind11 CONFIG REQUIRED HINTS "${pybind11_DIR}")
endif()

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module NumPy)

pybind11_add_module(orbit_integrator
  integrator/integrator.cpp
  integrator/bindings.cpp
)

target_include_directories(orbit_integrator PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/integrator
  ${Python_NumPy_INCLUDE_DIRS}
)

install(TARGETS orbit_integrator DESTINATION .)
