cmake_minimum_required(VERSION 3.0.2)
project(local3d_semantic_voxel_map)

add_compile_options(-std=c++14 -Wall -Wextra -Wpedantic)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  roscpp
  rospy
  sensor_msgs
  std_msgs
  std_srvs
  tf2
  tf2_ros
  tf2_sensor_msgs
  visualization_msgs
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES semantic_voxel_map
  CATKIN_DEPENDS
    geometry_msgs
    nav_msgs
    roscpp
    rospy
    sensor_msgs
    std_msgs
    std_srvs
    tf2
    tf2_ros
    tf2_sensor_msgs
    visualization_msgs
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_library(semantic_voxel_map
  src/global_semantic_admission.cpp
  src/semantic_fusion.cpp
  src/semantic_voxel_map.cpp
)
target_link_libraries(semantic_voxel_map ${catkin_LIBRARIES})
add_dependencies(semantic_voxel_map ${catkin_EXPORTED_TARGETS})

add_executable(local_3d_semantic_voxel_map_node
  src/semantic_voxel_map_node.cpp
)
target_link_libraries(local_3d_semantic_voxel_map_node
  semantic_voxel_map
  ${catkin_LIBRARIES}
)
add_dependencies(local_3d_semantic_voxel_map_node ${catkin_EXPORTED_TARGETS})

add_executable(test_local_goal_selector
  src/test_local_goal_selector.cpp
)
target_link_libraries(test_local_goal_selector
  ${catkin_LIBRARIES}
)
add_dependencies(test_local_goal_selector ${catkin_EXPORTED_TARGETS})

catkin_install_python(PROGRAMS
  scripts/record_path_goal_alignment.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

if(CATKIN_ENABLE_TESTING)
  catkin_add_gtest(test_semantic_voxel_map test/test_semantic_voxel_map.cpp)
  if(TARGET test_semantic_voxel_map)
    target_link_libraries(test_semantic_voxel_map semantic_voxel_map ${catkin_LIBRARIES})
  endif()
endif()

install(TARGETS semantic_voxel_map local_3d_semantic_voxel_map_node
  test_local_goal_selector
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

install(DIRECTORY config launch rviz
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(FILES README.md LOCAL_3D_VOXEL_COST_PIPELINE.md
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
