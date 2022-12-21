cmake_minimum_required(VERSION 3.22)
set(CMAKE_CXX_STANDARD 20)

project(hmm
    LANGUAGES CXX
    VERSION 0.0.1)

set(HMM_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(HMM_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(HMM_TEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/test")
set(HMM_WARNINGS -Wall -Wconversion -Wsign-conversion -Werror -Wextra -Wpedantic -Wpointer-arith)
set(HMM_SOURCES
    "${HMM_SOURCE_DIR}/hmm.cpp"
    "${HMM_INCLUDE_DIR}/hmm.h")

add_library(hmm
    STATIC "${HMM_SOURCES}")
target_include_directories(hmm
    PUBLIC  $<BUILD_INTERFACE:${HMM_INCLUDE_DIR}>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_options(hmm
    PRIVATE "${HMM_WARNINGS}")

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    option(HMM_BUILD_TESTS "Build the tests" On)

    if(HMM_BUILD_TESTS)
        add_executable(test
            "${HMM_TEST_DIR}/test.cpp")
        target_compile_options(test
            PRIVATE
            "${HMM_WARNINGS}")
        target_link_libraries(test
            PRIVATE hmm)
    endif()

    include(GNUInstallDirs)
    set(TARGETS_NAME ${PROJECT_NAME}Targets)
    set(INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")
    set(CONFIG_FILE_IN "${PROJECT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in")
    set(CONFIG_FILE_OUT "${PROJECT_BINARY_DIR}/cmake/${PROJECT_NAME}Config.cmake")
    set(VERSION_FILE "${PROJECT_BINARY_DIR}/cmake/${PROJECT_NAME}ConfigVersion.cmake")

    install(TARGETS hmm
        EXPORT "${TARGETS_NAME}"
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")
    install(FILES "${HMM_INCLUDE_DIR}/hmm.h"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

    include(CMakePackageConfigHelpers)
    configure_package_config_file(
        "${CONFIG_FILE_IN}" "${CONFIG_FILE_OUT}"
        INSTALL_DESTINATION "${INSTALL_DIR}")
    write_basic_package_version_file("${VERSION_FILE}"
        COMPATIBILITY SameMajorVersion)
    install(EXPORT "${TARGETS_NAME}"
        NAMESPACE "${PROJECT_NAME}::"
        DESTINATION "${INSTALL_DIR}")
    install(FILES "${CONFIG_FILE_OUT}" "${VERSION_FILE}"
        DESTINATION "${INSTALL_DIR}")
endif()