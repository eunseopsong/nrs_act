// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from vive_tracker_interfaces:srv/ViveCalibration.idl
// generated code does not contain a copyright notice

#ifndef VIVE_TRACKER_INTERFACES__SRV__DETAIL__VIVE_CALIBRATION__TRAITS_HPP_
#define VIVE_TRACKER_INTERFACES__SRV__DETAIL__VIVE_CALIBRATION__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "vive_tracker_interfaces/srv/detail/vive_calibration__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'robot_poses'
// Member 'tracker_poses'
#include "geometry_msgs/msg/detail/pose__traits.hpp"

namespace vive_tracker_interfaces
{

namespace srv
{

inline void to_flow_style_yaml(
  const ViveCalibration_Request & msg,
  std::ostream & out)
{
  out << "{";
  // member: robot_poses
  {
    if (msg.robot_poses.size() == 0) {
      out << "robot_poses: []";
    } else {
      out << "robot_poses: [";
      size_t pending_items = msg.robot_poses.size();
      for (auto item : msg.robot_poses) {
        to_flow_style_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: tracker_poses
  {
    if (msg.tracker_poses.size() == 0) {
      out << "tracker_poses: []";
    } else {
      out << "tracker_poses: [";
      size_t pending_items = msg.tracker_poses.size();
      for (auto item : msg.tracker_poses) {
        to_flow_style_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const ViveCalibration_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: robot_poses
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.robot_poses.size() == 0) {
      out << "robot_poses: []\n";
    } else {
      out << "robot_poses:\n";
      for (auto item : msg.robot_poses) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "-\n";
        to_block_style_yaml(item, out, indentation + 2);
      }
    }
  }

  // member: tracker_poses
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.tracker_poses.size() == 0) {
      out << "tracker_poses: []\n";
    } else {
      out << "tracker_poses:\n";
      for (auto item : msg.tracker_poses) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "-\n";
        to_block_style_yaml(item, out, indentation + 2);
      }
    }
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const ViveCalibration_Request & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace srv

}  // namespace vive_tracker_interfaces

namespace rosidl_generator_traits
{

[[deprecated("use vive_tracker_interfaces::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const vive_tracker_interfaces::srv::ViveCalibration_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  vive_tracker_interfaces::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use vive_tracker_interfaces::srv::to_yaml() instead")]]
inline std::string to_yaml(const vive_tracker_interfaces::srv::ViveCalibration_Request & msg)
{
  return vive_tracker_interfaces::srv::to_yaml(msg);
}

template<>
inline const char * data_type<vive_tracker_interfaces::srv::ViveCalibration_Request>()
{
  return "vive_tracker_interfaces::srv::ViveCalibration_Request";
}

template<>
inline const char * name<vive_tracker_interfaces::srv::ViveCalibration_Request>()
{
  return "vive_tracker_interfaces/srv/ViveCalibration_Request";
}

template<>
struct has_fixed_size<vive_tracker_interfaces::srv::ViveCalibration_Request>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<vive_tracker_interfaces::srv::ViveCalibration_Request>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<vive_tracker_interfaces::srv::ViveCalibration_Request>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace vive_tracker_interfaces
{

namespace srv
{

inline void to_flow_style_yaml(
  const ViveCalibration_Response & msg,
  std::ostream & out)
{
  out << "{";
  // member: success
  {
    out << "success: ";
    rosidl_generator_traits::value_to_yaml(msg.success, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const ViveCalibration_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: success
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "success: ";
    rosidl_generator_traits::value_to_yaml(msg.success, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const ViveCalibration_Response & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace srv

}  // namespace vive_tracker_interfaces

namespace rosidl_generator_traits
{

[[deprecated("use vive_tracker_interfaces::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const vive_tracker_interfaces::srv::ViveCalibration_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  vive_tracker_interfaces::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use vive_tracker_interfaces::srv::to_yaml() instead")]]
inline std::string to_yaml(const vive_tracker_interfaces::srv::ViveCalibration_Response & msg)
{
  return vive_tracker_interfaces::srv::to_yaml(msg);
}

template<>
inline const char * data_type<vive_tracker_interfaces::srv::ViveCalibration_Response>()
{
  return "vive_tracker_interfaces::srv::ViveCalibration_Response";
}

template<>
inline const char * name<vive_tracker_interfaces::srv::ViveCalibration_Response>()
{
  return "vive_tracker_interfaces/srv/ViveCalibration_Response";
}

template<>
struct has_fixed_size<vive_tracker_interfaces::srv::ViveCalibration_Response>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<vive_tracker_interfaces::srv::ViveCalibration_Response>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<vive_tracker_interfaces::srv::ViveCalibration_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<vive_tracker_interfaces::srv::ViveCalibration>()
{
  return "vive_tracker_interfaces::srv::ViveCalibration";
}

template<>
inline const char * name<vive_tracker_interfaces::srv::ViveCalibration>()
{
  return "vive_tracker_interfaces/srv/ViveCalibration";
}

template<>
struct has_fixed_size<vive_tracker_interfaces::srv::ViveCalibration>
  : std::integral_constant<
    bool,
    has_fixed_size<vive_tracker_interfaces::srv::ViveCalibration_Request>::value &&
    has_fixed_size<vive_tracker_interfaces::srv::ViveCalibration_Response>::value
  >
{
};

template<>
struct has_bounded_size<vive_tracker_interfaces::srv::ViveCalibration>
  : std::integral_constant<
    bool,
    has_bounded_size<vive_tracker_interfaces::srv::ViveCalibration_Request>::value &&
    has_bounded_size<vive_tracker_interfaces::srv::ViveCalibration_Response>::value
  >
{
};

template<>
struct is_service<vive_tracker_interfaces::srv::ViveCalibration>
  : std::true_type
{
};

template<>
struct is_service_request<vive_tracker_interfaces::srv::ViveCalibration_Request>
  : std::true_type
{
};

template<>
struct is_service_response<vive_tracker_interfaces::srv::ViveCalibration_Response>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits

#endif  // VIVE_TRACKER_INTERFACES__SRV__DETAIL__VIVE_CALIBRATION__TRAITS_HPP_
