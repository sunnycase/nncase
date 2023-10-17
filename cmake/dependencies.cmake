find_package(gsl-lite REQUIRED)
find_package(ncnn REQUIRED)

if (ENABLE_OPENMP)
    find_package(OpenMP COMPONENTS CXX REQUIRED)
endif ()

if ((NOT BUILDING_RUNTIME) OR ENABLE_VULKAN_RUNTIME)
    find_package(Vulkan REQUIRED)
endif ()

if (NOT BUILDING_RUNTIME)
    find_package(absl REQUIRED)
    find_package(nethost REQUIRED)
    find_package(fmt REQUIRED)
    find_package(magic_enum REQUIRED)
    find_package(spdlog REQUIRED)
    find_package(inja REQUIRED)
endif ()

if (BUILD_TESTING)
    find_package(GTest REQUIRED)
endif ()

if (ENABLE_HALIDE)
    find_package(hkg REQUIRED)
endif ()