#pragma once

#include <Ganymede/Source/Ganymede.h>

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/type_precision.hpp>

struct VertexAux
{
	glm::vec3 normal;
	glm::u8vec4 color;
};

struct ShaderConstants {
	glm::mat4 viewMtx;
	glm::mat4 projMtx;
	glm::mat4 viewInvMtx;
	glm::mat4 projInvMtx;
	glm::vec2 viewportSize;
	glm::vec2 viewportBase;
	uint32 numLights;
	uint32 numTriangles;
	uint32 frameIndex;
	uint32 numRays;
	uint32 numBVHNodes;
	alignas(16) glm::vec3 ambientRadiance;
};

struct Light
{
	alignas(16) glm::vec3 pos;
	alignas(16) glm::vec3 radiance;
};

struct RayStack
{
	alignas(16) glm::vec3 rayOrigin;
	alignas(16) glm::vec3 rayDirection;
	alignas(8) glm::u16vec4 hitAlbedo;
	alignas(8) glm::u16vec3 wIn;
};

struct RayJob
{
	uint32 index;
};