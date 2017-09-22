# ======================================================================
#  main build file
# ======================================================================

# use cmake 2.8 or later
CMAKE_MINIMUM_REQUIRED (VERSION 2.8)

# project name - use all lowercase
PROJECT(gm2fieldsignal)

# cetbuildtools contains our cmake modules
set( CETBUILDTOOLS_VERSION $ENV{CETBUILDTOOLS_VERSION} )
if( NOT CETBUILDTOOLS_VERSION )
  message( FATAL_ERROR "ERROR: setup cetbuildtools to get the cmake modules" )
endif()

set( CMAKE_MODULE_PATH $ENV{CANVAS_DIR}/Modules ${CMAKE_MODULE_PATH} )
set( CMAKE_MODULE_PATH $ENV{ART_DIR}/Modules ${CMAKE_MODULE_PATH} )
set( CMAKE_MODULE_PATH $ENV{CETBUILDTOOLS_DIR}/Modules ${CMAKE_MODULE_PATH} )

# include the necessary macros
include(CetCMakeEnv)
cet_cmake_env()

# g-2 wants the product name in the library name
set( ART_MAKE_PREPEND_PRODUCT_NAME TRUE)

# set compiler flags
cet_set_compiler_flags( DIAGS CAUTIOUS 
  WERROR
  EXTRA_FLAGS -pedantic
)

cet_report_compiler_flags()

# Locate some UPS areas (the versions are just minimum required) and setup include paths
# These are needed for Art stuff
find_ups_product(cetbuildtools v5_04_03)


# macros for art_make and simple plugins
#include(ArtMake)
#include(BuildPlugins)

# build source subdirectories here
add_subdirectory(ups)

# pckaging facility
include(UseCPack)
