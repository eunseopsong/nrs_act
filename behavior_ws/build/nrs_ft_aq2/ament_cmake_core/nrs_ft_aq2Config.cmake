# generated from ament/cmake/core/templates/nameConfig.cmake.in

# prevent multiple inclusion
if(_nrs_ft_aq2_CONFIG_INCLUDED)
  # ensure to keep the found flag the same
  if(NOT DEFINED nrs_ft_aq2_FOUND)
    # explicitly set it to FALSE, otherwise CMake will set it to TRUE
    set(nrs_ft_aq2_FOUND FALSE)
  elseif(NOT nrs_ft_aq2_FOUND)
    # use separate condition to avoid uninitialized variable warning
    set(nrs_ft_aq2_FOUND FALSE)
  endif()
  return()
endif()
set(_nrs_ft_aq2_CONFIG_INCLUDED TRUE)

# output package information
if(NOT nrs_ft_aq2_FIND_QUIETLY)
  message(STATUS "Found nrs_ft_aq2: 0.0.1 (${nrs_ft_aq2_DIR})")
endif()

# warn when using a deprecated package
if(NOT "" STREQUAL "")
  set(_msg "Package 'nrs_ft_aq2' is deprecated")
  # append custom deprecation text if available
  if(NOT "" STREQUAL "TRUE")
    set(_msg "${_msg} ()")
  endif()
  # optionally quiet the deprecation message
  if(NOT ${nrs_ft_aq2_DEPRECATED_QUIET})
    message(DEPRECATION "${_msg}")
  endif()
endif()

# flag package as ament-based to distinguish it after being find_package()-ed
set(nrs_ft_aq2_FOUND_AMENT_PACKAGE TRUE)

# include all config extra files
set(_extras "")
foreach(_extra ${_extras})
  include("${nrs_ft_aq2_DIR}/${_extra}")
endforeach()
