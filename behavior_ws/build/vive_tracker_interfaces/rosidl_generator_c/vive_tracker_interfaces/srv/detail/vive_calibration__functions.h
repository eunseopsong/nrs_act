// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from vive_tracker_interfaces:srv/ViveCalibration.idl
// generated code does not contain a copyright notice

#ifndef VIVE_TRACKER_INTERFACES__SRV__DETAIL__VIVE_CALIBRATION__FUNCTIONS_H_
#define VIVE_TRACKER_INTERFACES__SRV__DETAIL__VIVE_CALIBRATION__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_runtime_c/visibility_control.h"
#include "vive_tracker_interfaces/msg/rosidl_generator_c__visibility_control.h"

#include "vive_tracker_interfaces/srv/detail/vive_calibration__struct.h"

/// Initialize srv/ViveCalibration message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * vive_tracker_interfaces__srv__ViveCalibration_Request
 * )) before or use
 * vive_tracker_interfaces__srv__ViveCalibration_Request__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_vive_tracker_interfaces
bool
vive_tracker_interfaces__srv__ViveCalibration_Request__init(vive_tracker_interfaces__srv__ViveCalibration_Request * msg);

/// Finalize srv/ViveCalibration message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_vive_tracker_interfaces
void
vive_tracker_interfaces__srv__ViveCalibration_Request__fini(vive_tracker_interfaces__srv__ViveCalibration_Request * msg);

/// Create srv/ViveCalibration message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * vive_tracker_interfaces__srv__ViveCalibration_Request__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_vive_tracker_interfaces
vive_tracker_interfaces__srv__ViveCalibration_Request *
vive_tracker_interfaces__srv__ViveCalibration_Request__create();

/// Destroy srv/ViveCalibration message.
/**
 * It calls
 * vive_tracker_interfaces__srv__ViveCalibration_Request__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_vive_tracker_interfaces
void
vive_tracker_interfaces__srv__ViveCalibration_Request__destroy(vive_tracker_interfaces__srv__ViveCalibration_Request * msg);

/// Check for srv/ViveCalibration message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_vive_tracker_interfaces
bool
vive_tracker_interfaces__srv__ViveCalibration_Request__are_equal(const vive_tracker_interfaces__srv__ViveCalibration_Request * lhs, const vive_tracker_interfaces__srv__ViveCalibration_Request * rhs);

/// Copy a srv/ViveCalibration message.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source message pointer.
 * \param[out] output The target message pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer is null
 *   or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_vive_tracker_interfaces
bool
vive_tracker_interfaces__srv__ViveCalibration_Request__copy(
  const vive_tracker_interfaces__srv__ViveCalibration_Request * input,
  vive_tracker_interfaces__srv__ViveCalibration_Request * output);

/// Initialize array of srv/ViveCalibration messages.
/**
 * It allocates the memory for the number of elements and calls
 * vive_tracker_interfaces__srv__ViveCalibration_Request__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_vive_tracker_interfaces
bool
vive_tracker_interfaces__srv__ViveCalibration_Request__Sequence__init(vive_tracker_interfaces__srv__ViveCalibration_Request__Sequence * array, size_t size);

/// Finalize array of srv/ViveCalibration messages.
/**
 * It calls
 * vive_tracker_interfaces__srv__ViveCalibration_Request__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_vive_tracker_interfaces
void
vive_tracker_interfaces__srv__ViveCalibration_Request__Sequence__fini(vive_tracker_interfaces__srv__ViveCalibration_Request__Sequence * array);

/// Create array of srv/ViveCalibration messages.
/**
 * It allocates the memory for the array and calls
 * vive_tracker_interfaces__srv__ViveCalibration_Request__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_vive_tracker_interfaces
vive_tracker_interfaces__srv__ViveCalibration_Request__Sequence *
vive_tracker_interfaces__srv__ViveCalibration_Request__Sequence__create(size_t size);

/// Destroy array of srv/ViveCalibration messages.
/**
 * It calls
 * vive_tracker_interfaces__srv__ViveCalibration_Request__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_vive_tracker_interfaces
void
vive_tracker_interfaces__srv__ViveCalibration_Request__Sequence__destroy(vive_tracker_interfaces__srv__ViveCalibration_Request__Sequence * array);

/// Check for srv/ViveCalibration message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_vive_tracker_interfaces
bool
vive_tracker_interfaces__srv__ViveCalibration_Request__Sequence__are_equal(const vive_tracker_interfaces__srv__ViveCalibration_Request__Sequence * lhs, const vive_tracker_interfaces__srv__ViveCalibration_Request__Sequence * rhs);

/// Copy an array of srv/ViveCalibration messages.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source array pointer.
 * \param[out] output The target array pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer
 *   is null or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_vive_tracker_interfaces
bool
vive_tracker_interfaces__srv__ViveCalibration_Request__Sequence__copy(
  const vive_tracker_interfaces__srv__ViveCalibration_Request__Sequence * input,
  vive_tracker_interfaces__srv__ViveCalibration_Request__Sequence * output);

/// Initialize srv/ViveCalibration message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * vive_tracker_interfaces__srv__ViveCalibration_Response
 * )) before or use
 * vive_tracker_interfaces__srv__ViveCalibration_Response__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_vive_tracker_interfaces
bool
vive_tracker_interfaces__srv__ViveCalibration_Response__init(vive_tracker_interfaces__srv__ViveCalibration_Response * msg);

/// Finalize srv/ViveCalibration message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_vive_tracker_interfaces
void
vive_tracker_interfaces__srv__ViveCalibration_Response__fini(vive_tracker_interfaces__srv__ViveCalibration_Response * msg);

/// Create srv/ViveCalibration message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * vive_tracker_interfaces__srv__ViveCalibration_Response__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_vive_tracker_interfaces
vive_tracker_interfaces__srv__ViveCalibration_Response *
vive_tracker_interfaces__srv__ViveCalibration_Response__create();

/// Destroy srv/ViveCalibration message.
/**
 * It calls
 * vive_tracker_interfaces__srv__ViveCalibration_Response__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_vive_tracker_interfaces
void
vive_tracker_interfaces__srv__ViveCalibration_Response__destroy(vive_tracker_interfaces__srv__ViveCalibration_Response * msg);

/// Check for srv/ViveCalibration message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_vive_tracker_interfaces
bool
vive_tracker_interfaces__srv__ViveCalibration_Response__are_equal(const vive_tracker_interfaces__srv__ViveCalibration_Response * lhs, const vive_tracker_interfaces__srv__ViveCalibration_Response * rhs);

/// Copy a srv/ViveCalibration message.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source message pointer.
 * \param[out] output The target message pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer is null
 *   or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_vive_tracker_interfaces
bool
vive_tracker_interfaces__srv__ViveCalibration_Response__copy(
  const vive_tracker_interfaces__srv__ViveCalibration_Response * input,
  vive_tracker_interfaces__srv__ViveCalibration_Response * output);

/// Initialize array of srv/ViveCalibration messages.
/**
 * It allocates the memory for the number of elements and calls
 * vive_tracker_interfaces__srv__ViveCalibration_Response__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_vive_tracker_interfaces
bool
vive_tracker_interfaces__srv__ViveCalibration_Response__Sequence__init(vive_tracker_interfaces__srv__ViveCalibration_Response__Sequence * array, size_t size);

/// Finalize array of srv/ViveCalibration messages.
/**
 * It calls
 * vive_tracker_interfaces__srv__ViveCalibration_Response__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_vive_tracker_interfaces
void
vive_tracker_interfaces__srv__ViveCalibration_Response__Sequence__fini(vive_tracker_interfaces__srv__ViveCalibration_Response__Sequence * array);

/// Create array of srv/ViveCalibration messages.
/**
 * It allocates the memory for the array and calls
 * vive_tracker_interfaces__srv__ViveCalibration_Response__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_vive_tracker_interfaces
vive_tracker_interfaces__srv__ViveCalibration_Response__Sequence *
vive_tracker_interfaces__srv__ViveCalibration_Response__Sequence__create(size_t size);

/// Destroy array of srv/ViveCalibration messages.
/**
 * It calls
 * vive_tracker_interfaces__srv__ViveCalibration_Response__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_vive_tracker_interfaces
void
vive_tracker_interfaces__srv__ViveCalibration_Response__Sequence__destroy(vive_tracker_interfaces__srv__ViveCalibration_Response__Sequence * array);

/// Check for srv/ViveCalibration message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_vive_tracker_interfaces
bool
vive_tracker_interfaces__srv__ViveCalibration_Response__Sequence__are_equal(const vive_tracker_interfaces__srv__ViveCalibration_Response__Sequence * lhs, const vive_tracker_interfaces__srv__ViveCalibration_Response__Sequence * rhs);

/// Copy an array of srv/ViveCalibration messages.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source array pointer.
 * \param[out] output The target array pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer
 *   is null or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_vive_tracker_interfaces
bool
vive_tracker_interfaces__srv__ViveCalibration_Response__Sequence__copy(
  const vive_tracker_interfaces__srv__ViveCalibration_Response__Sequence * input,
  vive_tracker_interfaces__srv__ViveCalibration_Response__Sequence * output);

#ifdef __cplusplus
}
#endif

#endif  // VIVE_TRACKER_INTERFACES__SRV__DETAIL__VIVE_CALIBRATION__FUNCTIONS_H_
