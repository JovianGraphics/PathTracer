#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outPos;
layout(location = 2) out vec4 outNormal;

layout(location = 0) in vec4 fragColor;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 pos;

void main() {
    outColor = fragColor;
    outNormal = vec4(normal, 1.0);
    outPos = vec4(pos, 1.0);
}