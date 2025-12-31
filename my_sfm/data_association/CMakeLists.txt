ADD_LIBRARY(data_association
    data_association.cc
)

TARGET_INCLUDE_DIRECTORIES(data_association
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

TARGET_LINK_LIBRARIES(data_association
    PUBLIC data_structures database
)

ADD_EXECUTABLE(test1 test.cc)
TARGET_LINK_LIBRARIES(test1
    PRIVATE data_association database SQLite::SQLite3 ${OpenCV_LIBS} glog::glog
)
