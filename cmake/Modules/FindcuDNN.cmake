# Find the cuDNN libraries
#
# The following variables are optionally searched for defaults
#  cuDNN_ROOT: Base directory where cuDNN is found
#  cuDNN_INCLUDE_DIR: Directory where cuDNN header is searched for
#  cuDNN_LIBRARY: Directory where cuDNN library is searched for
#  cuDNN_STATIC: Are we looking for a static library? (default: no)
#
# The following are set after configuration is done:
#  cuDNN_FOUND
#  cuDNN_INCLUDE_PATH
#  cuDNN_LIBRARY_PATH
#

include(FindPackageHandleStandardArgs)

set(cuDNN_ROOT $ENV{CUDNN_ROOT_DIR} CACHE PATH "Folder containing NVIDIA cuDNN")
list(APPEND cuDNN_ROOT $ENV{CUDNN_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR})

set(cuDNN_INCLUDE_DIR $ENV{CUDNN_INCLUDE_DIR} CACHE PATH "Folder containing NVIDIA cuDNN header files")

find_path(cuDNN_INCLUDE_PATH cudnn.h
        HINTS ${cuDNN_INCLUDE_DIR}
        PATH_SUFFIXES cuda/include cuda include)

option(cuDNN_STATIC "Look for static cuDNN" OFF)
if (cuDNN_STATIC)
    set(cuDNN_LIBNAME "libcudnn_static.a")
else()
    set(cuDNN_LIBNAME "cudnn")
endif()

set(cuDNN_LIBRARY $ENV{cuDNN_LIBRARY} CACHE PATH "Path to the cudnn library file (e.g., libcudnn.so)")
if (cuDNN_LIBRARY MATCHES ".*cudnn_static.a" AND NOT cuDNN_STATIC)
    message(WARNING "cuDNN_LIBRARY points to a static library (${cuDNN_LIBRARY}) but cuDNN_STATIC is OFF.")
endif()

find_library(cuDNN_LIBRARY_PATH ${cuDNN_LIBNAME}
        PATHS ${cuDNN_LIBRARY}
        PATH_SUFFIXES lib lib64 cuda/lib cuda/lib64 lib/x64)

find_package_handle_standard_args(cuDNN DEFAULT_MSG cuDNN_LIBRARY_PATH cuDNN_INCLUDE_PATH)

if (cuDNN_STATIC)
    add_library(cuDNN STATIC IMPORTED)
else()
    add_library(cuDNN SHARED IMPORTED)
endif()
target_include_directories(cuDNN SYSTEM INTERFACE "${cuDNN_INCLUDE_PATH}")
set_target_properties(cuDNN PROPERTIES
        IMPORTED_LOCATION "${cuDNN_LIBRARY_PATH}"
        IMPORTED_IMPLIB   "${cuDNN_LIBRARY_PATH}/${cuDNN_LIBNAME}")

mark_as_advanced(cuDNN_ROOT cuDNN_INCLUDE_DIR cuDNN_LIBRARY)