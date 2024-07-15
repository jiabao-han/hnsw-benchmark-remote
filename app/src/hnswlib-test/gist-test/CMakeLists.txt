set(CONFIGURE_INCLUDE "${CMAKE_BINARY_DIR}/configured_files/include")

add_executable(gist-hnswlib-test "gist-hnswlib-test.cpp")

# Link the target against the libraries and frameworks
target_link_libraries(gist-hnswlib-test
        PUBLIC
        hnswlib::hnswlib
)

# Specify the directories for NGT include files
include_directories("${PROJECT_BINARY_DIR}/lib/")
include_directories("${CMAKE_BINARY_DIR}/configured_files/include")
