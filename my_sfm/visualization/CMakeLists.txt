ADD_LIBRARY(visualization
    visualization.cc
)

TARGET_INCLUDE_DIRECTORIES(visualization
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

TARGET_LINK_LIBRARIES(visualization
    PRIVATE ${OpenCV_LIBS} glog::glog
)
