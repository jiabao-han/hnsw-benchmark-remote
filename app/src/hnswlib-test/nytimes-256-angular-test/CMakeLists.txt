set(CONFIGURE_INCLUDE "${CMAKE_BINARY_DIR}/configured_files/include")

add_executable(nytimes-256-angular-hnswlib-test "nytimes-256-angular-hnswlib-test.cpp")

# Link the target against the libraries and frameworks
target_link_libraries(nytimes-256-angular-hnswlib-test
        PUBLIC
        hnswlib::hnswlib
)

# Specify the directories for NGT include files
include_directories("${PROJECT_BINARY_DIR}/lib/")
include_directories("${CMAKE_BINARY_DIR}/configured_files/include")
