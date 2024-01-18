cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(OPOL)

# set libraries postfix
set(CMAKE_RELEASE_POSTFIX "")
set(CMAKE_DEBUG_POSTFIX "-debug")

# try to find OPENCV
find_package(OpenCV REQUIRED)

if(NOT OpenCV_FOUND)
	message(FATAL_ERROR "Opencv not found.")
endif()

# try to find the eigen3 headers
find_package(Eigen3 REQUIRED)

# try to find the eigen3 headers
find_package(CUDA REQUIRED)

# Try to find OpenMP
find_package(OpenMP)

# try to find the boost headers
find_package(Boost REQUIRED COMPONENTS filesystem serialization program_options)

# find VCGLib
find_path(VCG_INCLUDE_DIR vcg/complex/base.h)

# find HNMSLib
find_path(HNSW_INCLUDE_DIR hnswlib.h)

#sqlite
find_path(OPOL_INCLUDE_DIR OPOL.h)
find_library(OPOL_LIBRARY_Release NAMES OPOL_Lib)
find_library(OPOL_LIBRARY_Debug NAMES OPOL_Lib)

# One-Point-One-Line Geometry
file( GLOB OPOL_header_list ${CMAKE_CURRENT_SOURCE_DIR}/OPOL/*.h ${CMAKE_CURRENT_SOURCE_DIR}/OPOL/*.hpp)
file( GLOB OPOL_source_list ${CMAKE_CURRENT_SOURCE_DIR}/OPOL/*.cpp) 
file( GLOB OPOL_cuda_list ${CMAKE_CURRENT_SOURCE_DIR}/OPOL/*.cu) 
cuda_add_executable(OPOL
		${OPOL_header_list}
		${OPOL_cuda_list}
		${OPOL_source_list})
target_include_directories(OPOL PUBLIC ${VCG_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
target_include_directories(OPOL PUBLIC ${EIGEN3_INCLUDE_DIRS})
target_include_directories(OPOL PUBLIC ${OPOL_INCLUDE_DIR})
target_include_directories(OPOL PUBLIC ${HNSW_INCLUDE_DIR})
target_link_libraries(OPOL debug ${OPOL_LIBRARY_Debug})
target_link_libraries(OPOL optimized ${OPOL_LIBRARY_Release})
target_link_libraries(OPOL ${OpenCV_LIBRARIES})
target_link_libraries(OPOL ${CUDA_cusolver_LIBRARY})
target_link_libraries(OPOL debug ${Boost_FILESYSTEM_LIBRARY_DEBUG})
target_link_libraries(OPOL optimized ${Boost_FILESYSTEM_LIBRARY_RELEASE})
target_link_libraries(OPOL debug ${Boost_SERIALIZATION_LIBRARY_DEBUG})
target_link_libraries(OPOL optimized ${Boost_SERIALIZATION_LIBRARY_RELEASE})

# Enable OpenMP if found
if(OpenMP_CXX_FOUND)
    target_compile_options(OPOL PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${OpenMP_CXX_FLAGS}>)
    target_link_libraries(OPOL PUBLIC OpenMP::OpenMP_CXX)
endif()


set_property(TARGET OPOL PROPERTY CXX_STANDARD 14)