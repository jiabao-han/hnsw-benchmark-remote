set(CONFIGURE_INCLUDE "${CMAKE_BINARY_DIR}/configured_files/include")

add_executable(nytimes-256-angular-n2-test "nytimes-256-angular-n2-test.cpp")

# Link the target against the libraries and frameworks
target_link_libraries(nytimes-256-angular-n2-test
        PUBLIC
)

# Specify the directories for NGT include files
include_directories("${PROJECT_BINARY_DIR}/lib/")
include_directories("${CMAKE_BINARY_DIR}/configured_files/include")
