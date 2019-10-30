file(GLOB CUR_SRC_FILES ${CMAKE_CURRENT_LIST_DIR}/*.cpp)
file(GLOB CUR_HDR_FILES ${CMAKE_CURRENT_LIST_DIR}/*.h)
set(SRC_FILES ${SRC_FILES} ${CUR_SRC_FILES})
set(HDR_FILES ${HDR_FILES} ${CUR_HDR_FILES})

file(GLOB CUR_SRC_FILES ${CMAKE_CURRENT_LIST_DIR}/graph/*.cpp)
file(GLOB CUR_HDR_FILES ${CMAKE_CURRENT_LIST_DIR}/graph/*.h)

list(REMOVE_ITEM CUR_SRC_FILES "${CMAKE_CURRENT_LIST_DIR}/graph/filterGraphSelector.cpp")

list(REMOVE_ITEM CUR_SRC_FILES "${CMAKE_CURRENT_LIST_DIR}/graph/diagramscene.cpp")
list(REMOVE_ITEM CUR_HDR_FILES "${CMAKE_CURRENT_LIST_DIR}/graph/diagramscene.h")



set(SRC_FILES ${SRC_FILES} ${CUR_SRC_FILES})
set(HDR_FILES ${HDR_FILES} ${CUR_HDR_FILES})

# file(GLOB CUR_SRC_FILES ${CMAKE_CURRENT_LIST_DIR}/legacy/*.cpp)
# file(GLOB CUR_HDR_FILES ${CMAKE_CURRENT_LIST_DIR}/legacy/*.h)
# set(SRC_FILES ${SRC_FILES} ${CUR_SRC_FILES})
# set(HDR_FILES ${HDR_FILES} ${CUR_HDR_FILES})

file(GLOB CUR_SRC_FILES ${CMAKE_CURRENT_LIST_DIR}/ui/*.cpp)
file(GLOB CUR_HDR_FILES ${CMAKE_CURRENT_LIST_DIR}/ui/*.h)
list(REMOVE_ITEM CUR_SRC_FILES "${CMAKE_CURRENT_LIST_DIR}/ui/operationFilterControlWidget.cpp")
set(SRC_FILES ${SRC_FILES} ${CUR_SRC_FILES})
set(HDR_FILES ${HDR_FILES} ${CUR_HDR_FILES})

include_directories(${CMAKE_CURRENT_LIST_DIR}/ui)


