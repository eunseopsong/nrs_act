// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from vive_tracker_interfaces:srv/ViveCalibration.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "vive_tracker_interfaces/srv/detail/vive_calibration__rosidl_typesupport_introspection_c.h"
#include "vive_tracker_interfaces/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "vive_tracker_interfaces/srv/detail/vive_calibration__functions.h"
#include "vive_tracker_interfaces/srv/detail/vive_calibration__struct.h"


// Include directives for member types
// Member `robot_poses`
// Member `tracker_poses`
#include "geometry_msgs/msg/pose.h"
// Member `robot_poses`
// Member `tracker_poses`
#include "geometry_msgs/msg/detail/pose__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void vive_tracker_interfaces__srv__ViveCalibration_Request__rosidl_typesupport_introspection_c__ViveCalibration_Request_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  vive_tracker_interfaces__srv__ViveCalibration_Request__init(message_memory);
}

void vive_tracker_interfaces__srv__ViveCalibration_Request__rosidl_typesupport_introspection_c__ViveCalibration_Request_fini_function(void * message_memory)
{
  vive_tracker_interfaces__srv__ViveCalibration_Request__fini(message_memory);
}

size_t vive_tracker_interfaces__srv__ViveCalibration_Request__rosidl_typesupport_introspection_c__size_function__ViveCalibration_Request__robot_poses(
  const void * untyped_member)
{
  const geometry_msgs__msg__Pose__Sequence * member =
    (const geometry_msgs__msg__Pose__Sequence *)(untyped_member);
  return member->size;
}

const void * vive_tracker_interfaces__srv__ViveCalibration_Request__rosidl_typesupport_introspection_c__get_const_function__ViveCalibration_Request__robot_poses(
  const void * untyped_member, size_t index)
{
  const geometry_msgs__msg__Pose__Sequence * member =
    (const geometry_msgs__msg__Pose__Sequence *)(untyped_member);
  return &member->data[index];
}

void * vive_tracker_interfaces__srv__ViveCalibration_Request__rosidl_typesupport_introspection_c__get_function__ViveCalibration_Request__robot_poses(
  void * untyped_member, size_t index)
{
  geometry_msgs__msg__Pose__Sequence * member =
    (geometry_msgs__msg__Pose__Sequence *)(untyped_member);
  return &member->data[index];
}

void vive_tracker_interfaces__srv__ViveCalibration_Request__rosidl_typesupport_introspection_c__fetch_function__ViveCalibration_Request__robot_poses(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const geometry_msgs__msg__Pose * item =
    ((const geometry_msgs__msg__Pose *)
    vive_tracker_interfaces__srv__ViveCalibration_Request__rosidl_typesupport_introspection_c__get_const_function__ViveCalibration_Request__robot_poses(untyped_member, index));
  geometry_msgs__msg__Pose * value =
    (geometry_msgs__msg__Pose *)(untyped_value);
  *value = *item;
}

void vive_tracker_interfaces__srv__ViveCalibration_Request__rosidl_typesupport_introspection_c__assign_function__ViveCalibration_Request__robot_poses(
  void * untyped_member, size_t index, const void * untyped_value)
{
  geometry_msgs__msg__Pose * item =
    ((geometry_msgs__msg__Pose *)
    vive_tracker_interfaces__srv__ViveCalibration_Request__rosidl_typesupport_introspection_c__get_function__ViveCalibration_Request__robot_poses(untyped_member, index));
  const geometry_msgs__msg__Pose * value =
    (const geometry_msgs__msg__Pose *)(untyped_value);
  *item = *value;
}

bool vive_tracker_interfaces__srv__ViveCalibration_Request__rosidl_typesupport_introspection_c__resize_function__ViveCalibration_Request__robot_poses(
  void * untyped_member, size_t size)
{
  geometry_msgs__msg__Pose__Sequence * member =
    (geometry_msgs__msg__Pose__Sequence *)(untyped_member);
  geometry_msgs__msg__Pose__Sequence__fini(member);
  return geometry_msgs__msg__Pose__Sequence__init(member, size);
}

size_t vive_tracker_interfaces__srv__ViveCalibration_Request__rosidl_typesupport_introspection_c__size_function__ViveCalibration_Request__tracker_poses(
  const void * untyped_member)
{
  const geometry_msgs__msg__Pose__Sequence * member =
    (const geometry_msgs__msg__Pose__Sequence *)(untyped_member);
  return member->size;
}

const void * vive_tracker_interfaces__srv__ViveCalibration_Request__rosidl_typesupport_introspection_c__get_const_function__ViveCalibration_Request__tracker_poses(
  const void * untyped_member, size_t index)
{
  const geometry_msgs__msg__Pose__Sequence * member =
    (const geometry_msgs__msg__Pose__Sequence *)(untyped_member);
  return &member->data[index];
}

void * vive_tracker_interfaces__srv__ViveCalibration_Request__rosidl_typesupport_introspection_c__get_function__ViveCalibration_Request__tracker_poses(
  void * untyped_member, size_t index)
{
  geometry_msgs__msg__Pose__Sequence * member =
    (geometry_msgs__msg__Pose__Sequence *)(untyped_member);
  return &member->data[index];
}

void vive_tracker_interfaces__srv__ViveCalibration_Request__rosidl_typesupport_introspection_c__fetch_function__ViveCalibration_Request__tracker_poses(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const geometry_msgs__msg__Pose * item =
    ((const geometry_msgs__msg__Pose *)
    vive_tracker_interfaces__srv__ViveCalibration_Request__rosidl_typesupport_introspection_c__get_const_function__ViveCalibration_Request__tracker_poses(untyped_member, index));
  geometry_msgs__msg__Pose * value =
    (geometry_msgs__msg__Pose *)(untyped_value);
  *value = *item;
}

void vive_tracker_interfaces__srv__ViveCalibration_Request__rosidl_typesupport_introspection_c__assign_function__ViveCalibration_Request__tracker_poses(
  void * untyped_member, size_t index, const void * untyped_value)
{
  geometry_msgs__msg__Pose * item =
    ((geometry_msgs__msg__Pose *)
    vive_tracker_interfaces__srv__ViveCalibration_Request__rosidl_typesupport_introspection_c__get_function__ViveCalibration_Request__tracker_poses(untyped_member, index));
  const geometry_msgs__msg__Pose * value =
    (const geometry_msgs__msg__Pose *)(untyped_value);
  *item = *value;
}

bool vive_tracker_interfaces__srv__ViveCalibration_Request__rosidl_typesupport_introspection_c__resize_function__ViveCalibration_Request__tracker_poses(
  void * untyped_member, size_t size)
{
  geometry_msgs__msg__Pose__Sequence * member =
    (geometry_msgs__msg__Pose__Sequence *)(untyped_member);
  geometry_msgs__msg__Pose__Sequence__fini(member);
  return geometry_msgs__msg__Pose__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember vive_tracker_interfaces__srv__ViveCalibration_Request__rosidl_typesupport_introspection_c__ViveCalibration_Request_message_member_array[2] = {
  {
    "robot_poses",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(vive_tracker_interfaces__srv__ViveCalibration_Request, robot_poses),  // bytes offset in struct
    NULL,  // default value
    vive_tracker_interfaces__srv__ViveCalibration_Request__rosidl_typesupport_introspection_c__size_function__ViveCalibration_Request__robot_poses,  // size() function pointer
    vive_tracker_interfaces__srv__ViveCalibration_Request__rosidl_typesupport_introspection_c__get_const_function__ViveCalibration_Request__robot_poses,  // get_const(index) function pointer
    vive_tracker_interfaces__srv__ViveCalibration_Request__rosidl_typesupport_introspection_c__get_function__ViveCalibration_Request__robot_poses,  // get(index) function pointer
    vive_tracker_interfaces__srv__ViveCalibration_Request__rosidl_typesupport_introspection_c__fetch_function__ViveCalibration_Request__robot_poses,  // fetch(index, &value) function pointer
    vive_tracker_interfaces__srv__ViveCalibration_Request__rosidl_typesupport_introspection_c__assign_function__ViveCalibration_Request__robot_poses,  // assign(index, value) function pointer
    vive_tracker_interfaces__srv__ViveCalibration_Request__rosidl_typesupport_introspection_c__resize_function__ViveCalibration_Request__robot_poses  // resize(index) function pointer
  },
  {
    "tracker_poses",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(vive_tracker_interfaces__srv__ViveCalibration_Request, tracker_poses),  // bytes offset in struct
    NULL,  // default value
    vive_tracker_interfaces__srv__ViveCalibration_Request__rosidl_typesupport_introspection_c__size_function__ViveCalibration_Request__tracker_poses,  // size() function pointer
    vive_tracker_interfaces__srv__ViveCalibration_Request__rosidl_typesupport_introspection_c__get_const_function__ViveCalibration_Request__tracker_poses,  // get_const(index) function pointer
    vive_tracker_interfaces__srv__ViveCalibration_Request__rosidl_typesupport_introspection_c__get_function__ViveCalibration_Request__tracker_poses,  // get(index) function pointer
    vive_tracker_interfaces__srv__ViveCalibration_Request__rosidl_typesupport_introspection_c__fetch_function__ViveCalibration_Request__tracker_poses,  // fetch(index, &value) function pointer
    vive_tracker_interfaces__srv__ViveCalibration_Request__rosidl_typesupport_introspection_c__assign_function__ViveCalibration_Request__tracker_poses,  // assign(index, value) function pointer
    vive_tracker_interfaces__srv__ViveCalibration_Request__rosidl_typesupport_introspection_c__resize_function__ViveCalibration_Request__tracker_poses  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers vive_tracker_interfaces__srv__ViveCalibration_Request__rosidl_typesupport_introspection_c__ViveCalibration_Request_message_members = {
  "vive_tracker_interfaces__srv",  // message namespace
  "ViveCalibration_Request",  // message name
  2,  // number of fields
  sizeof(vive_tracker_interfaces__srv__ViveCalibration_Request),
  vive_tracker_interfaces__srv__ViveCalibration_Request__rosidl_typesupport_introspection_c__ViveCalibration_Request_message_member_array,  // message members
  vive_tracker_interfaces__srv__ViveCalibration_Request__rosidl_typesupport_introspection_c__ViveCalibration_Request_init_function,  // function to initialize message memory (memory has to be allocated)
  vive_tracker_interfaces__srv__ViveCalibration_Request__rosidl_typesupport_introspection_c__ViveCalibration_Request_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t vive_tracker_interfaces__srv__ViveCalibration_Request__rosidl_typesupport_introspection_c__ViveCalibration_Request_message_type_support_handle = {
  0,
  &vive_tracker_interfaces__srv__ViveCalibration_Request__rosidl_typesupport_introspection_c__ViveCalibration_Request_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_vive_tracker_interfaces
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, vive_tracker_interfaces, srv, ViveCalibration_Request)() {
  vive_tracker_interfaces__srv__ViveCalibration_Request__rosidl_typesupport_introspection_c__ViveCalibration_Request_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, geometry_msgs, msg, Pose)();
  vive_tracker_interfaces__srv__ViveCalibration_Request__rosidl_typesupport_introspection_c__ViveCalibration_Request_message_member_array[1].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, geometry_msgs, msg, Pose)();
  if (!vive_tracker_interfaces__srv__ViveCalibration_Request__rosidl_typesupport_introspection_c__ViveCalibration_Request_message_type_support_handle.typesupport_identifier) {
    vive_tracker_interfaces__srv__ViveCalibration_Request__rosidl_typesupport_introspection_c__ViveCalibration_Request_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &vive_tracker_interfaces__srv__ViveCalibration_Request__rosidl_typesupport_introspection_c__ViveCalibration_Request_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

// already included above
// #include <stddef.h>
// already included above
// #include "vive_tracker_interfaces/srv/detail/vive_calibration__rosidl_typesupport_introspection_c.h"
// already included above
// #include "vive_tracker_interfaces/msg/rosidl_typesupport_introspection_c__visibility_control.h"
// already included above
// #include "rosidl_typesupport_introspection_c/field_types.h"
// already included above
// #include "rosidl_typesupport_introspection_c/identifier.h"
// already included above
// #include "rosidl_typesupport_introspection_c/message_introspection.h"
// already included above
// #include "vive_tracker_interfaces/srv/detail/vive_calibration__functions.h"
// already included above
// #include "vive_tracker_interfaces/srv/detail/vive_calibration__struct.h"


#ifdef __cplusplus
extern "C"
{
#endif

void vive_tracker_interfaces__srv__ViveCalibration_Response__rosidl_typesupport_introspection_c__ViveCalibration_Response_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  vive_tracker_interfaces__srv__ViveCalibration_Response__init(message_memory);
}

void vive_tracker_interfaces__srv__ViveCalibration_Response__rosidl_typesupport_introspection_c__ViveCalibration_Response_fini_function(void * message_memory)
{
  vive_tracker_interfaces__srv__ViveCalibration_Response__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember vive_tracker_interfaces__srv__ViveCalibration_Response__rosidl_typesupport_introspection_c__ViveCalibration_Response_message_member_array[1] = {
  {
    "success",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(vive_tracker_interfaces__srv__ViveCalibration_Response, success),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers vive_tracker_interfaces__srv__ViveCalibration_Response__rosidl_typesupport_introspection_c__ViveCalibration_Response_message_members = {
  "vive_tracker_interfaces__srv",  // message namespace
  "ViveCalibration_Response",  // message name
  1,  // number of fields
  sizeof(vive_tracker_interfaces__srv__ViveCalibration_Response),
  vive_tracker_interfaces__srv__ViveCalibration_Response__rosidl_typesupport_introspection_c__ViveCalibration_Response_message_member_array,  // message members
  vive_tracker_interfaces__srv__ViveCalibration_Response__rosidl_typesupport_introspection_c__ViveCalibration_Response_init_function,  // function to initialize message memory (memory has to be allocated)
  vive_tracker_interfaces__srv__ViveCalibration_Response__rosidl_typesupport_introspection_c__ViveCalibration_Response_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t vive_tracker_interfaces__srv__ViveCalibration_Response__rosidl_typesupport_introspection_c__ViveCalibration_Response_message_type_support_handle = {
  0,
  &vive_tracker_interfaces__srv__ViveCalibration_Response__rosidl_typesupport_introspection_c__ViveCalibration_Response_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_vive_tracker_interfaces
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, vive_tracker_interfaces, srv, ViveCalibration_Response)() {
  if (!vive_tracker_interfaces__srv__ViveCalibration_Response__rosidl_typesupport_introspection_c__ViveCalibration_Response_message_type_support_handle.typesupport_identifier) {
    vive_tracker_interfaces__srv__ViveCalibration_Response__rosidl_typesupport_introspection_c__ViveCalibration_Response_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &vive_tracker_interfaces__srv__ViveCalibration_Response__rosidl_typesupport_introspection_c__ViveCalibration_Response_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

#include "rosidl_runtime_c/service_type_support_struct.h"
// already included above
// #include "vive_tracker_interfaces/msg/rosidl_typesupport_introspection_c__visibility_control.h"
// already included above
// #include "vive_tracker_interfaces/srv/detail/vive_calibration__rosidl_typesupport_introspection_c.h"
// already included above
// #include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/service_introspection.h"

// this is intentionally not const to allow initialization later to prevent an initialization race
static rosidl_typesupport_introspection_c__ServiceMembers vive_tracker_interfaces__srv__detail__vive_calibration__rosidl_typesupport_introspection_c__ViveCalibration_service_members = {
  "vive_tracker_interfaces__srv",  // service namespace
  "ViveCalibration",  // service name
  // these two fields are initialized below on the first access
  NULL,  // request message
  // vive_tracker_interfaces__srv__detail__vive_calibration__rosidl_typesupport_introspection_c__ViveCalibration_Request_message_type_support_handle,
  NULL  // response message
  // vive_tracker_interfaces__srv__detail__vive_calibration__rosidl_typesupport_introspection_c__ViveCalibration_Response_message_type_support_handle
};

static rosidl_service_type_support_t vive_tracker_interfaces__srv__detail__vive_calibration__rosidl_typesupport_introspection_c__ViveCalibration_service_type_support_handle = {
  0,
  &vive_tracker_interfaces__srv__detail__vive_calibration__rosidl_typesupport_introspection_c__ViveCalibration_service_members,
  get_service_typesupport_handle_function,
};

// Forward declaration of request/response type support functions
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, vive_tracker_interfaces, srv, ViveCalibration_Request)();

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, vive_tracker_interfaces, srv, ViveCalibration_Response)();

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_vive_tracker_interfaces
const rosidl_service_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_introspection_c, vive_tracker_interfaces, srv, ViveCalibration)() {
  if (!vive_tracker_interfaces__srv__detail__vive_calibration__rosidl_typesupport_introspection_c__ViveCalibration_service_type_support_handle.typesupport_identifier) {
    vive_tracker_interfaces__srv__detail__vive_calibration__rosidl_typesupport_introspection_c__ViveCalibration_service_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  rosidl_typesupport_introspection_c__ServiceMembers * service_members =
    (rosidl_typesupport_introspection_c__ServiceMembers *)vive_tracker_interfaces__srv__detail__vive_calibration__rosidl_typesupport_introspection_c__ViveCalibration_service_type_support_handle.data;

  if (!service_members->request_members_) {
    service_members->request_members_ =
      (const rosidl_typesupport_introspection_c__MessageMembers *)
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, vive_tracker_interfaces, srv, ViveCalibration_Request)()->data;
  }
  if (!service_members->response_members_) {
    service_members->response_members_ =
      (const rosidl_typesupport_introspection_c__MessageMembers *)
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, vive_tracker_interfaces, srv, ViveCalibration_Response)()->data;
  }

  return &vive_tracker_interfaces__srv__detail__vive_calibration__rosidl_typesupport_introspection_c__ViveCalibration_service_type_support_handle;
}
