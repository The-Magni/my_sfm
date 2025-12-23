ADD_LIBRARY(data_structures
    scene_graph.cc
    images.cc
    thread_pool.cc
    camera.cc
    point_cloud.cc
)

TARGET_INCLUDE_DIRECTORIES(data_structures
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

ADD_EXECUTABLE(test test.cc)
TARGET_LINK_LIBRARIES(test PRIVATE data_structures ${OpenCV_LIBS})
