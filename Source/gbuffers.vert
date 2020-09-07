#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_shader_16bit_storage : enable
#extension GL_EXT_shader_explicit_arithmetic_types : enable

layout(binding = 0) uniform Constants {
    mat4 viewMtx;
    mat4 projMtx;
    mat4 viewInvMtx;
    mat4 projInvMtx;
    vec2 viewportSize;
    uint numLights;
    uint numTriangles;
    uint frameIndex;
    uint numRays;
    uint numBVHNodes;
};

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec4 inColor;
layout(location = 2) in vec3 inNormal;

layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec3 normal;
layout(location = 2) out vec3 pos;

layout(std430, binding = 4) buffer blueNoiseBuffer
{
    uint16_t blueNoise[];
};

#include "noise.glsl"

void main() {
    pos = inPosition;
    fragColor = inColor;
    normal = inNormal;

    gl_Position = projMtx * (viewMtx * vec4(inPosition, 1.0));

    int jitter = blueNoise[frameIndex];

    gl_Position.st += (WeylNth(jitter) * 2.0 - 1.0) / viewportSize;
}