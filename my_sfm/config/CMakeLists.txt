ADD_LIBRARY(config
    config.cc
)

TARGET_INCLUDE_DIRECTORIES(config
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

TARGET_LINK_LIBRARIES(config
    PRIVATE glog::glog yaml-cpp
)
