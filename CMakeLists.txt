cmake_minimum_required(VERSION 3.0.2)
project(local3d_semantic_voxel_map)

add_compile_options(-std=c++17 -Wall -Wextra -Wpedantic)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  roscpp
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
    roscpp
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

if(CATKIN_ENABLE_TESTING)
  catkin_add_gtest(test_semantic_voxel_map test/test_semantic_voxel_map.cpp)
  if(TARGET test_semantic_voxel_map)
    target_link_libraries(test_semantic_voxel_map semantic_voxel_map ${catkin_LIBRARIES})
  endif()
endif()

install(TARGETS semantic_voxel_map local_3d_semantic_voxel_map_node
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
