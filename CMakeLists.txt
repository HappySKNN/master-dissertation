cmake_minimum_required(VERSION 3.10)
project(dissertation)

set(CMAKE_CXX_STANDARD 17)
set(PYTHON_EXECUTABLE "/usr/bin/python3")

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

include_directories(
        ${Python3_INCLUDE_DIRS}
        "C:/Users/skyfox/AppData/Local/Programs/Python/Python313/Lib/site-packages/numpy/_core/include"
        Include
        Datasets
        "C:/Eigen"
)

add_executable(dissertation main.cpp
        Include/FFBS.h
        Include/MCMC.h
        Utility/CSVLoader.h
        Utility/Graphics.h
        Source/MCMC.cpp
        Source/FFBS.cpp
)

target_link_libraries(dissertation
        ${Python3_LIBRARIES}
)
