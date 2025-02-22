cmake_minimum_required(VERSION 3.10)

project(Test)

file(GLOB_RECURSE SRC ./src/*.cpp)

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set (CMAKE_CXX_STANDARD 17)

find_package(Vulkan)
find_package(VulkanMemoryAllocator CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(Stb REQUIRED)

add_executable(test.out ${SRC})
target_link_libraries(test.out PRIVATE Vulkan::Vulkan GPUOpen::VulkanMemoryAllocator)
target_link_libraries(test.out PRIVATE glfw)
target_include_directories(test.out PUBLIC "../../Viz")
target_include_directories(test.out PRIVATE ${Stb_INCLUDE_DIR})
