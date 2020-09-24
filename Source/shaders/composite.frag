#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_shader_16bit_storage : enable
#extension GL_EXT_shader_explicit_arithmetic_types : enable

layout(location = 0) out vec3 outColor;

#include "structures.glsl"

layout(binding = 5, rgba16f) uniform image2D currentImage;
layout(binding = 6, rgba32f) uniform image2D accumulation;

layout(std430, binding = 9) buffer stackBuffer
{
    RayStackBuffer rayStack[];
};

vec3 ACESFilm(vec3 x)
{
    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;
    return clamp((x*(a*x+b))/(x*(c*x+d)+e), vec3(0.0), vec3(1.0));
}

layout(location = 0) in vec2 screenUV;

void main()
{
    uvec2 fragmentLoc = uvec2(screenUV * viewportSize);
    uint stackIndex = (fragmentLoc.y * uint(viewportSize.x) + fragmentLoc.x) * numRays;

    vec3 L = vec3(0.0);
    for (int depth = int(numRays) - 1; depth >= 0; depth--)
    {
        float prob = float(rayStack[stackIndex + depth].prob);
        if (prob > 0.0)
        {
            L *= rayStack[stackIndex + depth].prob;
            L += rayStack[stackIndex + depth].wIn;
            L *= rayStack[stackIndex + depth].hitAlbedo.rgb; // hemisphere samples
        }
        else
        {
            L = vec3(0.0);
        }
    }

    imageStore(currentImage, ivec2(gl_FragCoord.st), vec4(L, 1.0));

    vec4 acc = imageLoad(accumulation, ivec2(gl_FragCoord.st));
    acc += vec4(L, 1.0);
    imageStore(accumulation, ivec2(gl_FragCoord.st), acc);

    outColor = pow(ACESFilm(acc.rgb / acc.w), vec3(1.0 / 2.2));
}