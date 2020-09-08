#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) out vec3 outColor;

layout(binding = 6, rgba32f) uniform image2D currAccumulation;

vec3 ACESFilm(vec3 x)
{
    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;
    return clamp((x*(a*x+b))/(x*(c*x+d)+e), vec3(0.0), vec3(1.0));
}

void main()
{
    vec4 acc = imageLoad(currAccumulation, ivec2(gl_FragCoord.st));
    outColor = ACESFilm(acc.rgb / acc.w);
}