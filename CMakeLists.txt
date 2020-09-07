# CMakeList.txt : CMake project for JovianGames, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project ("PathTracer")

include_directories(${CMAKE_BINARY_DIR}/generated)

add_subdirectory(JovianGraphics)

if (WIN32)
	add_executable(PathTracer WIN32
		Source/PathTracer.cpp
		Source/BVH.cpp
	)
	target_sources(PathTracer PRIVATE ${CMAKE_SOURCE_DIR}/JovianGraphics/Io/WindowsHiDPI.manifest)
else()
	add_executable(PathTracer
		Source/PathTracer.cpp
		Source/BVH.cpp
	)
endif()

add_custom_command(
	OUTPUT trace.frag.h
	PRE_BUILD
	COMMAND ruby ${CMAKE_SOURCE_DIR}/JovianGraphics/Europa/Tools/compile_shader.rb ${CMAKE_SOURCE_DIR}/Source/trace.frag output=${CMAKE_BINARY_DIR}/generated/trace.frag.h
	MAIN_DEPENDENCY ${CMAKE_SOURCE_DIR}/Source/trace.frag
)

add_custom_command(
	OUTPUT trace.vert.h
	PRE_BUILD
	COMMAND ruby ${CMAKE_SOURCE_DIR}/JovianGraphics/Europa/Tools/compile_shader.rb ${CMAKE_SOURCE_DIR}/Source/trace.vert output=${CMAKE_BINARY_DIR}/generated/trace.vert.h
	MAIN_DEPENDENCY ${CMAKE_SOURCE_DIR}/Source/trace.vert
)

add_custom_command(
	OUTPUT visualize.frag.h
	PRE_BUILD
	COMMAND ruby ${CMAKE_SOURCE_DIR}/JovianGraphics/Europa/Tools/compile_shader.rb ${CMAKE_SOURCE_DIR}/Source/visualize.frag output=${CMAKE_BINARY_DIR}/generated/visualize.frag.h
	MAIN_DEPENDENCY ${CMAKE_SOURCE_DIR}/Source/visualize.frag
)

add_custom_command(
	OUTPUT visualize.vert.h
	PRE_BUILD
	COMMAND ruby ${CMAKE_SOURCE_DIR}/JovianGraphics/Europa/Tools/compile_shader.rb ${CMAKE_SOURCE_DIR}/Source/visualize.vert output=${CMAKE_BINARY_DIR}/generated/visualize.vert.h
	MAIN_DEPENDENCY ${CMAKE_SOURCE_DIR}/Source/visualize.vert
)

add_custom_command(
	OUTPUT gbuffers.frag.h
	PRE_BUILD
	COMMAND ruby ${CMAKE_SOURCE_DIR}/JovianGraphics/Europa/Tools/compile_shader.rb ${CMAKE_SOURCE_DIR}/Source/gbuffers.frag output=${CMAKE_BINARY_DIR}/generated/gbuffers.frag.h
	MAIN_DEPENDENCY ${CMAKE_SOURCE_DIR}/Source/gbuffers.frag
)

add_custom_command(
	OUTPUT gbuffers.vert.h
	PRE_BUILD
	COMMAND ruby ${CMAKE_SOURCE_DIR}/JovianGraphics/Europa/Tools/compile_shader.rb ${CMAKE_SOURCE_DIR}/Source/gbuffers.vert output=${CMAKE_BINARY_DIR}/generated/gbuffers.vert.h
	MAIN_DEPENDENCY ${CMAKE_SOURCE_DIR}/Source/gbuffers.vert
)

add_custom_target(Shaders ALL DEPENDS
	trace.frag.h
	trace.vert.h
	visualize.frag.h
	visualize.vert.h
	gbuffers.frag.h
	gbuffers.vert.h
)

add_dependencies(PathTracer Shaders)

set_property(TARGET PathTracer PROPERTY CXX_STANDARD 17)

target_link_libraries(PathTracer PUBLIC Amalthea Io Europa Himalia Ganymede)
target_include_directories(PathTracer PUBLIC ${JovianIncludeDir} Source)

include_directories(Source)
