// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from vive_tracker_interfaces:srv/ViveCalibration.idl
// generated code does not contain a copyright notice

#ifndef VIVE_TRACKER_INTERFACES__SRV__DETAIL__VIVE_CALIBRATION__STRUCT_H_
#define VIVE_TRACKER_INTERFACES__SRV__DETAIL__VIVE_CALIBRATION__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'robot_poses'
// Member 'tracker_poses'
#include "geometry_msgs/msg/detail/pose__struct.h"

/// Struct defined in srv/ViveCalibration in the package vive_tracker_interfaces.
typedef struct vive_tracker_interfaces__srv__ViveCalibration_Request
{
  geometry_msgs__msg__Pose__Sequence robot_poses;
  geometry_msgs__msg__Pose__Sequence tracker_poses;
} vive_tracker_interfaces__srv__ViveCalibration_Request;

// Struct for a sequence of vive_tracker_interfaces__srv__ViveCalibration_Request.
typedef struct vive_tracker_interfaces__srv__ViveCalibration_Request__Sequence
{
  vive_tracker_interfaces__srv__ViveCalibration_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} vive_tracker_interfaces__srv__ViveCalibration_Request__Sequence;


// Constants defined in the message

/// Struct defined in srv/ViveCalibration in the package vive_tracker_interfaces.
typedef struct vive_tracker_interfaces__srv__ViveCalibration_Response
{
  bool success;
} vive_tracker_interfaces__srv__ViveCalibration_Response;

// Struct for a sequence of vive_tracker_interfaces__srv__ViveCalibration_Response.
typedef struct vive_tracker_interfaces__srv__ViveCalibration_Response__Sequence
{
  vive_tracker_interfaces__srv__ViveCalibration_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} vive_tracker_interfaces__srv__ViveCalibration_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // VIVE_TRACKER_INTERFACES__SRV__DETAIL__VIVE_CALIBRATION__STRUCT_H_
