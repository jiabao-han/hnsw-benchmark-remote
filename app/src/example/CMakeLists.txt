set(CONFIGURE_INCLUDE "${CMAKE_BINARY_DIR}/configured_files/include")

set(CMAKE_CXX_STANDARD 23)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    SET( CMAKE_CXX_FLAGS  "-Ofast -std=c++23 -DHAVE_CXX0X -openmp -fpic -ftree-vectorize" )
    check_cxx_compiler_flag("-march=native" COMPILER_SUPPORT_NATIVE_FLAG)
    if(COMPILER_SUPPORT_NATIVE_FLAG)
        SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native" )
        message("set -march=native flag")
    else()
        check_cxx_compiler_flag("-mcpu=apple-m1" COMPILER_SUPPORT_M1_FLAG)
        if(COMPILER_SUPPORT_M1_FLAG)
            SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=apple-m1" )
            message("set -mcpu=apple-m1 flag")
        endif()
    endif()
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    SET( CMAKE_CXX_FLAGS  "-Ofast -lrt -std=c++23 -DHAVE_CXX0X -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0" )
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    SET( CMAKE_CXX_FLAGS  "/O2 -DHAVE_CXX0X /W1 /openmp /EHsc" )
endif()

add_executable(example_search ${CMAKE_CURRENT_SOURCE_DIR}/example_search.cpp)
target_link_libraries(example_search hnswlib)
#target_include_directories(example_search PUBLIC ${CONFIGURE_INCLUDE})

add_executable(example_epsilon_search ${CMAKE_CURRENT_SOURCE_DIR}/example_epsilon_search.cpp)
target_link_libraries(example_epsilon_search hnswlib)

add_executable(example_multivector_search ${CMAKE_CURRENT_SOURCE_DIR}/example_multivector_search.cpp)
target_link_libraries(example_multivector_search hnswlib)

add_executable(example_filter ${CMAKE_CURRENT_SOURCE_DIR}/example_filter.cpp)
target_link_libraries(example_filter hnswlib)

add_executable(example_replace_deleted ${CMAKE_CURRENT_SOURCE_DIR}/example_replace_deleted.cpp)
target_link_libraries(example_replace_deleted hnswlib)

add_executable(example_mt_search ${CMAKE_CURRENT_SOURCE_DIR}/example_mt_search.cpp)
target_link_libraries(example_mt_search hnswlib)

add_executable(example_mt_filter ${CMAKE_CURRENT_SOURCE_DIR}/example_mt_filter.cpp)
target_link_libraries(example_mt_filter hnswlib)

add_executable(example_mt_replace_deleted ${CMAKE_CURRENT_SOURCE_DIR}/example_mt_replace_deleted.cpp)
target_link_libraries(example_mt_replace_deleted hnswlib)

# tests
add_executable(multivector_search_test ${CMAKE_CURRENT_SOURCE_DIR}/multivector_search_test.cpp)
target_link_libraries(multivector_search_test hnswlib)

add_executable(epsilon_search_test ${CMAKE_CURRENT_SOURCE_DIR}/epsilon_search_test.cpp)
target_link_libraries(epsilon_search_test hnswlib)

add_executable(test_updates ${CMAKE_CURRENT_SOURCE_DIR}/updates_test.cpp)
target_link_libraries(test_updates hnswlib)
target_include_directories(test_updates PUBLIC ${CONFIGURE_INCLUDE})

add_executable(sift_test ${CMAKE_CURRENT_SOURCE_DIR}/sift_test.cpp)
target_link_libraries(sift_test hnswlib)
target_include_directories(sift_test PUBLIC ${CONFIGURE_INCLUDE})

add_executable(searchKnnCloserFirst_test ${CMAKE_CURRENT_SOURCE_DIR}/searchKnnCloserFirst_test.cpp)
target_link_libraries(searchKnnCloserFirst_test hnswlib)

add_executable(searchKnnWithFilter_test ${CMAKE_CURRENT_SOURCE_DIR}/searchKnnWithFilter_test.cpp)
target_link_libraries(searchKnnWithFilter_test hnswlib)

add_executable(multiThreadLoad_test ${CMAKE_CURRENT_SOURCE_DIR}/multiThreadLoad_test.cpp)
target_link_libraries(multiThreadLoad_test hnswlib)

add_executable(multiThread_replace_test ${CMAKE_CURRENT_SOURCE_DIR}/multiThread_replace_test.cpp)
target_link_libraries(multiThread_replace_test hnswlib)

