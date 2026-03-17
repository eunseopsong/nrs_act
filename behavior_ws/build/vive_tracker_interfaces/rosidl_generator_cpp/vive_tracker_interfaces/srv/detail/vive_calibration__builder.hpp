// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from vive_tracker_interfaces:srv/ViveCalibration.idl
// generated code does not contain a copyright notice

#ifndef VIVE_TRACKER_INTERFACES__SRV__DETAIL__VIVE_CALIBRATION__BUILDER_HPP_
#define VIVE_TRACKER_INTERFACES__SRV__DETAIL__VIVE_CALIBRATION__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "vive_tracker_interfaces/srv/detail/vive_calibration__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace vive_tracker_interfaces
{

namespace srv
{

namespace builder
{

class Init_ViveCalibration_Request_tracker_poses
{
public:
  explicit Init_ViveCalibration_Request_tracker_poses(::vive_tracker_interfaces::srv::ViveCalibration_Request & msg)
  : msg_(msg)
  {}
  ::vive_tracker_interfaces::srv::ViveCalibration_Request tracker_poses(::vive_tracker_interfaces::srv::ViveCalibration_Request::_tracker_poses_type arg)
  {
    msg_.tracker_poses = std::move(arg);
    return std::move(msg_);
  }

private:
  ::vive_tracker_interfaces::srv::ViveCalibration_Request msg_;
};

class Init_ViveCalibration_Request_robot_poses
{
public:
  Init_ViveCalibration_Request_robot_poses()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_ViveCalibration_Request_tracker_poses robot_poses(::vive_tracker_interfaces::srv::ViveCalibration_Request::_robot_poses_type arg)
  {
    msg_.robot_poses = std::move(arg);
    return Init_ViveCalibration_Request_tracker_poses(msg_);
  }

private:
  ::vive_tracker_interfaces::srv::ViveCalibration_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::vive_tracker_interfaces::srv::ViveCalibration_Request>()
{
  return vive_tracker_interfaces::srv::builder::Init_ViveCalibration_Request_robot_poses();
}

}  // namespace vive_tracker_interfaces


namespace vive_tracker_interfaces
{

namespace srv
{

namespace builder
{

class Init_ViveCalibration_Response_success
{
public:
  Init_ViveCalibration_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::vive_tracker_interfaces::srv::ViveCalibration_Response success(::vive_tracker_interfaces::srv::ViveCalibration_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::vive_tracker_interfaces::srv::ViveCalibration_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::vive_tracker_interfaces::srv::ViveCalibration_Response>()
{
  return vive_tracker_interfaces::srv::builder::Init_ViveCalibration_Response_success();
}

}  // namespace vive_tracker_interfaces

#endif  // VIVE_TRACKER_INTERFACES__SRV__DETAIL__VIVE_CALIBRATION__BUILDER_HPP_
