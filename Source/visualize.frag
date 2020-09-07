#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) out vec3 outColor;

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec4 pos;

void main() {
    outColor = fragColor;
}