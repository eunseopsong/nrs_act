// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from vive_tracker_interfaces:srv/ViveCalibration.idl
// generated code does not contain a copyright notice
#include "vive_tracker_interfaces/srv/detail/vive_calibration__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"

// Include directives for member types
// Member `robot_poses`
// Member `tracker_poses`
#include "geometry_msgs/msg/detail/pose__functions.h"

bool
vive_tracker_interfaces__srv__ViveCalibration_Request__init(vive_tracker_interfaces__srv__ViveCalibration_Request * msg)
{
  if (!msg) {
    return false;
  }
  // robot_poses
  if (!geometry_msgs__msg__Pose__Sequence__init(&msg->robot_poses, 0)) {
    vive_tracker_interfaces__srv__ViveCalibration_Request__fini(msg);
    return false;
  }
  // tracker_poses
  if (!geometry_msgs__msg__Pose__Sequence__init(&msg->tracker_poses, 0)) {
    vive_tracker_interfaces__srv__ViveCalibration_Request__fini(msg);
    return false;
  }
  return true;
}

void
vive_tracker_interfaces__srv__ViveCalibration_Request__fini(vive_tracker_interfaces__srv__ViveCalibration_Request * msg)
{
  if (!msg) {
    return;
  }
  // robot_poses
  geometry_msgs__msg__Pose__Sequence__fini(&msg->robot_poses);
  // tracker_poses
  geometry_msgs__msg__Pose__Sequence__fini(&msg->tracker_poses);
}

bool
vive_tracker_interfaces__srv__ViveCalibration_Request__are_equal(const vive_tracker_interfaces__srv__ViveCalibration_Request * lhs, const vive_tracker_interfaces__srv__ViveCalibration_Request * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // robot_poses
  if (!geometry_msgs__msg__Pose__Sequence__are_equal(
      &(lhs->robot_poses), &(rhs->robot_poses)))
  {
    return false;
  }
  // tracker_poses
  if (!geometry_msgs__msg__Pose__Sequence__are_equal(
      &(lhs->tracker_poses), &(rhs->tracker_poses)))
  {
    return false;
  }
  return true;
}

bool
vive_tracker_interfaces__srv__ViveCalibration_Request__copy(
  const vive_tracker_interfaces__srv__ViveCalibration_Request * input,
  vive_tracker_interfaces__srv__ViveCalibration_Request * output)
{
  if (!input || !output) {
    return false;
  }
  // robot_poses
  if (!geometry_msgs__msg__Pose__Sequence__copy(
      &(input->robot_poses), &(output->robot_poses)))
  {
    return false;
  }
  // tracker_poses
  if (!geometry_msgs__msg__Pose__Sequence__copy(
      &(input->tracker_poses), &(output->tracker_poses)))
  {
    return false;
  }
  return true;
}

vive_tracker_interfaces__srv__ViveCalibration_Request *
vive_tracker_interfaces__srv__ViveCalibration_Request__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  vive_tracker_interfaces__srv__ViveCalibration_Request * msg = (vive_tracker_interfaces__srv__ViveCalibration_Request *)allocator.allocate(sizeof(vive_tracker_interfaces__srv__ViveCalibration_Request), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(vive_tracker_interfaces__srv__ViveCalibration_Request));
  bool success = vive_tracker_interfaces__srv__ViveCalibration_Request__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
vive_tracker_interfaces__srv__ViveCalibration_Request__destroy(vive_tracker_interfaces__srv__ViveCalibration_Request * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    vive_tracker_interfaces__srv__ViveCalibration_Request__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
vive_tracker_interfaces__srv__ViveCalibration_Request__Sequence__init(vive_tracker_interfaces__srv__ViveCalibration_Request__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  vive_tracker_interfaces__srv__ViveCalibration_Request * data = NULL;

  if (size) {
    data = (vive_tracker_interfaces__srv__ViveCalibration_Request *)allocator.zero_allocate(size, sizeof(vive_tracker_interfaces__srv__ViveCalibration_Request), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = vive_tracker_interfaces__srv__ViveCalibration_Request__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        vive_tracker_interfaces__srv__ViveCalibration_Request__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
vive_tracker_interfaces__srv__ViveCalibration_Request__Sequence__fini(vive_tracker_interfaces__srv__ViveCalibration_Request__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      vive_tracker_interfaces__srv__ViveCalibration_Request__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

vive_tracker_interfaces__srv__ViveCalibration_Request__Sequence *
vive_tracker_interfaces__srv__ViveCalibration_Request__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  vive_tracker_interfaces__srv__ViveCalibration_Request__Sequence * array = (vive_tracker_interfaces__srv__ViveCalibration_Request__Sequence *)allocator.allocate(sizeof(vive_tracker_interfaces__srv__ViveCalibration_Request__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = vive_tracker_interfaces__srv__ViveCalibration_Request__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
vive_tracker_interfaces__srv__ViveCalibration_Request__Sequence__destroy(vive_tracker_interfaces__srv__ViveCalibration_Request__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    vive_tracker_interfaces__srv__ViveCalibration_Request__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
vive_tracker_interfaces__srv__ViveCalibration_Request__Sequence__are_equal(const vive_tracker_interfaces__srv__ViveCalibration_Request__Sequence * lhs, const vive_tracker_interfaces__srv__ViveCalibration_Request__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!vive_tracker_interfaces__srv__ViveCalibration_Request__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
vive_tracker_interfaces__srv__ViveCalibration_Request__Sequence__copy(
  const vive_tracker_interfaces__srv__ViveCalibration_Request__Sequence * input,
  vive_tracker_interfaces__srv__ViveCalibration_Request__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(vive_tracker_interfaces__srv__ViveCalibration_Request);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    vive_tracker_interfaces__srv__ViveCalibration_Request * data =
      (vive_tracker_interfaces__srv__ViveCalibration_Request *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!vive_tracker_interfaces__srv__ViveCalibration_Request__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          vive_tracker_interfaces__srv__ViveCalibration_Request__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!vive_tracker_interfaces__srv__ViveCalibration_Request__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}


bool
vive_tracker_interfaces__srv__ViveCalibration_Response__init(vive_tracker_interfaces__srv__ViveCalibration_Response * msg)
{
  if (!msg) {
    return false;
  }
  // success
  return true;
}

void
vive_tracker_interfaces__srv__ViveCalibration_Response__fini(vive_tracker_interfaces__srv__ViveCalibration_Response * msg)
{
  if (!msg) {
    return;
  }
  // success
}

bool
vive_tracker_interfaces__srv__ViveCalibration_Response__are_equal(const vive_tracker_interfaces__srv__ViveCalibration_Response * lhs, const vive_tracker_interfaces__srv__ViveCalibration_Response * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // success
  if (lhs->success != rhs->success) {
    return false;
  }
  return true;
}

bool
vive_tracker_interfaces__srv__ViveCalibration_Response__copy(
  const vive_tracker_interfaces__srv__ViveCalibration_Response * input,
  vive_tracker_interfaces__srv__ViveCalibration_Response * output)
{
  if (!input || !output) {
    return false;
  }
  // success
  output->success = input->success;
  return true;
}

vive_tracker_interfaces__srv__ViveCalibration_Response *
vive_tracker_interfaces__srv__ViveCalibration_Response__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  vive_tracker_interfaces__srv__ViveCalibration_Response * msg = (vive_tracker_interfaces__srv__ViveCalibration_Response *)allocator.allocate(sizeof(vive_tracker_interfaces__srv__ViveCalibration_Response), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(vive_tracker_interfaces__srv__ViveCalibration_Response));
  bool success = vive_tracker_interfaces__srv__ViveCalibration_Response__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
vive_tracker_interfaces__srv__ViveCalibration_Response__destroy(vive_tracker_interfaces__srv__ViveCalibration_Response * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    vive_tracker_interfaces__srv__ViveCalibration_Response__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
vive_tracker_interfaces__srv__ViveCalibration_Response__Sequence__init(vive_tracker_interfaces__srv__ViveCalibration_Response__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  vive_tracker_interfaces__srv__ViveCalibration_Response * data = NULL;

  if (size) {
    data = (vive_tracker_interfaces__srv__ViveCalibration_Response *)allocator.zero_allocate(size, sizeof(vive_tracker_interfaces__srv__ViveCalibration_Response), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = vive_tracker_interfaces__srv__ViveCalibration_Response__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        vive_tracker_interfaces__srv__ViveCalibration_Response__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
vive_tracker_interfaces__srv__ViveCalibration_Response__Sequence__fini(vive_tracker_interfaces__srv__ViveCalibration_Response__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      vive_tracker_interfaces__srv__ViveCalibration_Response__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

vive_tracker_interfaces__srv__ViveCalibration_Response__Sequence *
vive_tracker_interfaces__srv__ViveCalibration_Response__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  vive_tracker_interfaces__srv__ViveCalibration_Response__Sequence * array = (vive_tracker_interfaces__srv__ViveCalibration_Response__Sequence *)allocator.allocate(sizeof(vive_tracker_interfaces__srv__ViveCalibration_Response__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = vive_tracker_interfaces__srv__ViveCalibration_Response__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
vive_tracker_interfaces__srv__ViveCalibration_Response__Sequence__destroy(vive_tracker_interfaces__srv__ViveCalibration_Response__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    vive_tracker_interfaces__srv__ViveCalibration_Response__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
vive_tracker_interfaces__srv__ViveCalibration_Response__Sequence__are_equal(const vive_tracker_interfaces__srv__ViveCalibration_Response__Sequence * lhs, const vive_tracker_interfaces__srv__ViveCalibration_Response__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!vive_tracker_interfaces__srv__ViveCalibration_Response__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
vive_tracker_interfaces__srv__ViveCalibration_Response__Sequence__copy(
  const vive_tracker_interfaces__srv__ViveCalibration_Response__Sequence * input,
  vive_tracker_interfaces__srv__ViveCalibration_Response__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(vive_tracker_interfaces__srv__ViveCalibration_Response);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    vive_tracker_interfaces__srv__ViveCalibration_Response * data =
      (vive_tracker_interfaces__srv__ViveCalibration_Response *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!vive_tracker_interfaces__srv__ViveCalibration_Response__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          vive_tracker_interfaces__srv__ViveCalibration_Response__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!vive_tracker_interfaces__srv__ViveCalibration_Response__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
