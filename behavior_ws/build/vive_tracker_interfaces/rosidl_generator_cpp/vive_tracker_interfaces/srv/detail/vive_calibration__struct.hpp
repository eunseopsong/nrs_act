// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from vive_tracker_interfaces:srv/ViveCalibration.idl
// generated code does not contain a copyright notice

#ifndef VIVE_TRACKER_INTERFACES__SRV__DETAIL__VIVE_CALIBRATION__STRUCT_HPP_
#define VIVE_TRACKER_INTERFACES__SRV__DETAIL__VIVE_CALIBRATION__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'robot_poses'
// Member 'tracker_poses'
#include "geometry_msgs/msg/detail/pose__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__vive_tracker_interfaces__srv__ViveCalibration_Request __attribute__((deprecated))
#else
# define DEPRECATED__vive_tracker_interfaces__srv__ViveCalibration_Request __declspec(deprecated)
#endif

namespace vive_tracker_interfaces
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct ViveCalibration_Request_
{
  using Type = ViveCalibration_Request_<ContainerAllocator>;

  explicit ViveCalibration_Request_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_init;
  }

  explicit ViveCalibration_Request_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_init;
    (void)_alloc;
  }

  // field types and members
  using _robot_poses_type =
    std::vector<geometry_msgs::msg::Pose_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<geometry_msgs::msg::Pose_<ContainerAllocator>>>;
  _robot_poses_type robot_poses;
  using _tracker_poses_type =
    std::vector<geometry_msgs::msg::Pose_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<geometry_msgs::msg::Pose_<ContainerAllocator>>>;
  _tracker_poses_type tracker_poses;

  // setters for named parameter idiom
  Type & set__robot_poses(
    const std::vector<geometry_msgs::msg::Pose_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<geometry_msgs::msg::Pose_<ContainerAllocator>>> & _arg)
  {
    this->robot_poses = _arg;
    return *this;
  }
  Type & set__tracker_poses(
    const std::vector<geometry_msgs::msg::Pose_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<geometry_msgs::msg::Pose_<ContainerAllocator>>> & _arg)
  {
    this->tracker_poses = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    vive_tracker_interfaces::srv::ViveCalibration_Request_<ContainerAllocator> *;
  using ConstRawPtr =
    const vive_tracker_interfaces::srv::ViveCalibration_Request_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<vive_tracker_interfaces::srv::ViveCalibration_Request_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<vive_tracker_interfaces::srv::ViveCalibration_Request_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      vive_tracker_interfaces::srv::ViveCalibration_Request_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<vive_tracker_interfaces::srv::ViveCalibration_Request_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      vive_tracker_interfaces::srv::ViveCalibration_Request_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<vive_tracker_interfaces::srv::ViveCalibration_Request_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<vive_tracker_interfaces::srv::ViveCalibration_Request_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<vive_tracker_interfaces::srv::ViveCalibration_Request_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__vive_tracker_interfaces__srv__ViveCalibration_Request
    std::shared_ptr<vive_tracker_interfaces::srv::ViveCalibration_Request_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__vive_tracker_interfaces__srv__ViveCalibration_Request
    std::shared_ptr<vive_tracker_interfaces::srv::ViveCalibration_Request_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const ViveCalibration_Request_ & other) const
  {
    if (this->robot_poses != other.robot_poses) {
      return false;
    }
    if (this->tracker_poses != other.tracker_poses) {
      return false;
    }
    return true;
  }
  bool operator!=(const ViveCalibration_Request_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct ViveCalibration_Request_

// alias to use template instance with default allocator
using ViveCalibration_Request =
  vive_tracker_interfaces::srv::ViveCalibration_Request_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace vive_tracker_interfaces


#ifndef _WIN32
# define DEPRECATED__vive_tracker_interfaces__srv__ViveCalibration_Response __attribute__((deprecated))
#else
# define DEPRECATED__vive_tracker_interfaces__srv__ViveCalibration_Response __declspec(deprecated)
#endif

namespace vive_tracker_interfaces
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct ViveCalibration_Response_
{
  using Type = ViveCalibration_Response_<ContainerAllocator>;

  explicit ViveCalibration_Response_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->success = false;
    }
  }

  explicit ViveCalibration_Response_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->success = false;
    }
  }

  // field types and members
  using _success_type =
    bool;
  _success_type success;

  // setters for named parameter idiom
  Type & set__success(
    const bool & _arg)
  {
    this->success = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    vive_tracker_interfaces::srv::ViveCalibration_Response_<ContainerAllocator> *;
  using ConstRawPtr =
    const vive_tracker_interfaces::srv::ViveCalibration_Response_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<vive_tracker_interfaces::srv::ViveCalibration_Response_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<vive_tracker_interfaces::srv::ViveCalibration_Response_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      vive_tracker_interfaces::srv::ViveCalibration_Response_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<vive_tracker_interfaces::srv::ViveCalibration_Response_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      vive_tracker_interfaces::srv::ViveCalibration_Response_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<vive_tracker_interfaces::srv::ViveCalibration_Response_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<vive_tracker_interfaces::srv::ViveCalibration_Response_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<vive_tracker_interfaces::srv::ViveCalibration_Response_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__vive_tracker_interfaces__srv__ViveCalibration_Response
    std::shared_ptr<vive_tracker_interfaces::srv::ViveCalibration_Response_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__vive_tracker_interfaces__srv__ViveCalibration_Response
    std::shared_ptr<vive_tracker_interfaces::srv::ViveCalibration_Response_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const ViveCalibration_Response_ & other) const
  {
    if (this->success != other.success) {
      return false;
    }
    return true;
  }
  bool operator!=(const ViveCalibration_Response_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct ViveCalibration_Response_

// alias to use template instance with default allocator
using ViveCalibration_Response =
  vive_tracker_interfaces::srv::ViveCalibration_Response_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace vive_tracker_interfaces

namespace vive_tracker_interfaces
{

namespace srv
{

struct ViveCalibration
{
  using Request = vive_tracker_interfaces::srv::ViveCalibration_Request;
  using Response = vive_tracker_interfaces::srv::ViveCalibration_Response;
};

}  // namespace srv

}  // namespace vive_tracker_interfaces

#endif  // VIVE_TRACKER_INTERFACES__SRV__DETAIL__VIVE_CALIBRATION__STRUCT_HPP_
