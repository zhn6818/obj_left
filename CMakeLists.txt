cmake_minimum_required(VERSION 2.8)
project(ObjLeft)

cmake_policy(SET CMP0054 NEW)
set(CMAKE_CXX_STANDARD 11)

if(MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    #add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

if (CMAKE_HOST_WIN32)
set(WINDOWS 1)
elseif (CMAKE_HOST_APPLE)
set(MACOS 1)
elseif (CMAKE_HOST_UNIX)
set(LINUX 1)
endif ()

if(MACOS)
    message("this is mac os package")
    find_package(OpenCV REQUIRED)
endif(MACOS)


if(WINDOWS)
    message("this is windows cmake")
    set(OpenCV_DIR "D:\\opencv\\opencv\\build")
    set(OpenCV_INCLUDE_DIRS "D:\\opencv\\opencv\\build\\include")
    set( OpenCV_FOUND 1 )
    include_directories( ${OpenCV_INCLUDE_DIRS} )
    find_package(OpenCV REQUIRED)
    find_package(CUDA REQUIRED)
endif(WINDOWS)


link_directories(${MYPROJECT_DIR})

add_definitions(-O2 -g)

#set(CMAKE_C_COMPILER "/usr/bin/gcc") #这里写你的gcc路径
#set(CMAKE_CXX_COMPILER "/usr/bin/g++") #这里写你的g++路径
#set(OPENMP_LIBRARIES "/usr/local/lib") #这里写你的libomp路径, 通过brew安装的默认地址
#set(OPENMP_INCLUDES "/usr/local/include")#这里写你的libomp路径

#message("here")
#message("${CMAKE_CXX_COMPILER}")

#OPTION (USE_OpenMP "Use OpenMP to enamble <omp.h>" FALSE)

# Find OpenMP
#if(APPLE AND USE_OpenMP)
#    if(CMAKE_C_COMPILER_ID MATCHES "Clang")
#        set(OpenMP_C "${CMAKE_C_COMPILER}")
#        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -lomp -Wno-unused-command-line-argument")
#        #注意以上需要增加-Xpreprocessor 以及不能直接-llibomp 在这里不需要前缀lib只需要-lomp即可，下面相似的地方也是同个道理
#        set(OpenMP_C_LIB_NAMES "libomp" "libgomp" "libiomp5")
#        set(OpenMP_libomp_LIBRARY ${OpenMP_C_LIB_NAMES})
#        set(OpenMP_libgomp_LIBRARY ${OpenMP_C_LIB_NAMES})
#        set(OpenMP_libiomp5_LIBRARY ${OpenMP_C_LIB_NAMES})
#    endif()
#    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
#        set(OpenMP_CXX "${CMAKE_CXX_COMPILER}")
#        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -lomp -Wno-unused-command-line-argument")
#        set(OpenMP_CXX_LIB_NAMES "libomp" "libgomp" "libiomp5")
#        set(OpenMP_libomp_LIBRARY ${OpenMP_CXX_LIB_NAMES})
#        set(OpenMP_libgomp_LIBRARY ${OpenMP_CXX_LIB_NAMES})
#        set(OpenMP_libiomp5_LIBRARY ${OpenMP_CXX_LIB_NAMES})
#    endif()
#endif()
#
#if(USE_OpenMP)
#    message("opencvMp")
#    find_package(OpenMP REQUIRED)
#endif(USE_OpenMP)
#
## Search for OpenMP support for multi-threading
#if (OPENMP_FOUND)
#    message("OMP found")
#    message(${OPENMP_INCLUDES})
#    include_directories("${OPENMP_INCLUDES}")
#    link_directories("${OPENMP_LIBRARIES}")
#    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#endif(OPENMP_FOUND)


#set (CMAKE_CXX_FLAGS "-O2 -march=native")
include_directories(${CUDA_INCLUDE_DIRS})
if(MSVC)
cuda_add_executable(ObjLeft main CBM_model myDIP myGMM myUtility ObjLeftDetect VideoDetails test.cu test.h)
endif()
if(MACOS)
    add_executable(ObjLeft main CBM_model myDIP myGMM myUtility ObjLeftDetect VideoDetails)
endif()
#target_include_directories(ObjLeft
#        ${OpenCV_INCLUDE_DIRS}
#        ${CUDA_TOOLKIT_ROOT_DIR}/include)
target_link_libraries( ObjLeft ${OpenCV_LIBS})
