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
	OUTPUT trace.comp.h
	PRE_BUILD
	COMMAND ruby ${CMAKE_SOURCE_DIR}/JovianGraphics/Europa/Tools/compile_shader.rb ${CMAKE_SOURCE_DIR}/Source/shaders/trace.comp output=${CMAKE_BINARY_DIR}/generated/trace.comp.h
	MAIN_DEPENDENCY ${CMAKE_SOURCE_DIR}/Source/shaders/trace.comp
)

add_custom_command(
	OUTPUT launch.comp.h
	PRE_BUILD
	COMMAND ruby ${CMAKE_SOURCE_DIR}/JovianGraphics/Europa/Tools/compile_shader.rb ${CMAKE_SOURCE_DIR}/Source/shaders/launch.comp output=${CMAKE_BINARY_DIR}/generated/launch.comp.h
	MAIN_DEPENDENCY ${CMAKE_SOURCE_DIR}/Source/shaders/launch.comp
)

add_custom_command(
	OUTPUT raysort.comp.h
	PRE_BUILD
	COMMAND ruby ${CMAKE_SOURCE_DIR}/JovianGraphics/Europa/Tools/compile_shader.rb ${CMAKE_SOURCE_DIR}/Source/shaders/raysort.comp output=${CMAKE_BINARY_DIR}/generated/raysort.comp.h
	MAIN_DEPENDENCY ${CMAKE_SOURCE_DIR}/Source/shaders/raysort.comp
)

add_custom_command(
	OUTPUT visualize.frag.h
	PRE_BUILD
	COMMAND ruby ${CMAKE_SOURCE_DIR}/JovianGraphics/Europa/Tools/compile_shader.rb ${CMAKE_SOURCE_DIR}/Source/shaders/visualize.frag output=${CMAKE_BINARY_DIR}/generated/visualize.frag.h
	MAIN_DEPENDENCY ${CMAKE_SOURCE_DIR}/Source/shaders/visualize.frag
)

add_custom_command(
	OUTPUT visualize.vert.h
	PRE_BUILD
	COMMAND ruby ${CMAKE_SOURCE_DIR}/JovianGraphics/Europa/Tools/compile_shader.rb ${CMAKE_SOURCE_DIR}/Source/shaders/visualize.vert output=${CMAKE_BINARY_DIR}/generated/visualize.vert.h
	MAIN_DEPENDENCY ${CMAKE_SOURCE_DIR}/Source/shaders/visualize.vert
)

add_custom_command(
	OUTPUT composite.frag.h
	PRE_BUILD
	COMMAND ruby ${CMAKE_SOURCE_DIR}/JovianGraphics/Europa/Tools/compile_shader.rb ${CMAKE_SOURCE_DIR}/Source/shaders/composite.frag output=${CMAKE_BINARY_DIR}/generated/composite.frag.h
	MAIN_DEPENDENCY ${CMAKE_SOURCE_DIR}/Source/shaders/composite.frag
)

add_custom_command(
	OUTPUT composite.vert.h
	PRE_BUILD
	COMMAND ruby ${CMAKE_SOURCE_DIR}/JovianGraphics/Europa/Tools/compile_shader.rb ${CMAKE_SOURCE_DIR}/Source/shaders/composite.vert output=${CMAKE_BINARY_DIR}/generated/composite.vert.h
	MAIN_DEPENDENCY ${CMAKE_SOURCE_DIR}/Source/shaders/composite.vert
)

add_custom_target(Shaders ALL DEPENDS
	trace.comp.h
	launch.comp.h
	raysort.comp.h
	visualize.frag.h
	visualize.vert.h
	composite.frag.h
	composite.vert.h
)

add_dependencies(PathTracer Shaders)

set_property(TARGET PathTracer PROPERTY CXX_STANDARD 17)

target_link_libraries(PathTracer PUBLIC Amalthea Io Europa Himalia Ganymede)
target_include_directories(PathTracer PUBLIC ${JovianIncludeDir} Source)

include_directories(Source)
