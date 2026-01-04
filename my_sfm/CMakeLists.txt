ADD_SUBDIRECTORY(data_structures)
ADD_SUBDIRECTORY(data_association)
ADD_SUBDIRECTORY(database)
ADD_SUBDIRECTORY(reconstruction)
ADD_SUBDIRECTORY(visualization)

ADD_EXECUTABLE(main main.cc)
TARGET_LINK_LIBRARIES(main PRIVATE data_association data_structures database reconstruction visualization ${OpenCV_LIBS})
