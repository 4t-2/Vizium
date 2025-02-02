cmake_minimum_required(VERSION 3.10)

project(Test)

file(GLOB_RECURSE SRC ./src/*.cpp)

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set (CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_COMPILER "g++")

# set(CMAKE_BUILD_TYPE Debug) # default build

message(STATUS "CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")

if(CMAKE_BUILD_TYPE MATCHES Debug)
	set (CMAKE_CXX_FLAGS "-g -fsanitize=address")
	message(STATUS "Building with debug flags")
elseif(CMAKE_BUILD_TYPE MATCHES Release)
	set (CMAKE_CXX_FLAGS "-Ofast -flto=auto")
	message(STATUS "Building with release flags")
endif()

add_executable(a ${SRC})
target_link_libraries(a -lglfw -lvulkan)

#target_include_directories(EvolutionSimulator INTERFACE )
#link_libraries(${GLEW_LIBRARIES})
#target_include_directories(EvolutionSimulator )
