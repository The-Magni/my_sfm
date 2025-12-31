ADD_LIBRARY(database
    keypoints.cc
    descriptors.cc
    two_view_geometries.cc
)

TARGET_INCLUDE_DIRECTORIES(database
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

ADD_EXECUTABLE(test2 test.cc)
TARGET_LINK_LIBRARIES(database
    PUBLIC SQLite::SQLite3
)
TARGET_LINK_LIBRARIES(test2 PRIVATE database SQLite::SQLite3 glog::glog)
