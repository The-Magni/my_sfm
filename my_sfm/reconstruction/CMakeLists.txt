ADD_LIBRARY( reconstruction
    reconstruction.cc
    optimization.cc
    outlier_filtering.cc
)

TARGET_INCLUDE_DIRECTORIES(reconstruction
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

TARGET_LINK_LIBRARIES(reconstruction
    PUBLIC data_structures database Ceres::ceres Eigen3::Eigen
)

ADD_EXECUTABLE(test3 test.cc)
TARGET_LINK_LIBRARIES(test3
    PRIVATE reconstruction database Ceres::ceres glog::glog
)
