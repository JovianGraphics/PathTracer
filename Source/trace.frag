#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_shader_16bit_storage : enable

layout(binding = 0) uniform Constants {
    mat4 viewMtx;
    mat4 projMtx;
    mat4 viewInvMtx;
    mat4 projInvMtx;
    uint numLights;
    uint numTriangles;
};

layout(std430) struct Light
{
    vec3 pos;
    vec3 radiance;
};

layout(std430, binding = 1) buffer lightBuffer
{
    Light lights[];
};

layout(binding = 2) buffer vertexBuffer
{
    vec3 vertices[];
};

layout(binding = 3) buffer indexBuffer
{
    uint16_t indicies[];
};

layout(location = 0) in vec2 screenUV;

layout(location = 0) out vec4 outColor;

#include "noise.glsl"

struct Ray
{
  vec3 o;
  vec3 d;
  float min_t;
  float max_t;
};

struct Triangle
{
  int i1, i2, i3;
  vec3 p1, p2, p3;
  vec3 n1, n2, n3;
  vec3 c1, c2, c3;
};

struct Intersection
{
  vec3 bary;
  vec3 n;
  vec3 c;
  float t;
};

bool intersect(inout Ray r, Triangle tri, inout Intersection isect)
{
  vec3 e1 = tri.p2 - tri.p1;
  vec3 e2 = tri.p3 - tri.p1;
  vec3 s = r.o - tri.p1, s1 = cross(r.d, e2), s2 = cross(s, e1);
  vec3 matrix = vec3(dot(s2, e2), dot(s1, s), dot(s2, r.d));
  vec3 intersection = matrix / dot(s1, e1);

  float t = intersection.x;
  float alpha = intersection.y;
  float beta = intersection.z;
  float gamma = 1.0 - alpha - beta;

  if (t < r.min_t || t > r.max_t || alpha < 0 || beta < 0 || gamma < 0 || gamma > 1) return false;

  r.max_t = t;

  tri.c1 = vertices[tri.i1 * 3 + 1].xyz;
  tri.c2 = vertices[tri.i2 * 3 + 1].xyz;
  tri.c3 = vertices[tri.i3 * 3 + 1].xyz;

  tri.n1 = vertices[tri.i1 * 3 + 2].xyz;
  tri.n2 = vertices[tri.i2 * 3 + 2].xyz;
  tri.n3 = vertices[tri.i3 * 3 + 2].xyz;

  isect.bary = vec3(gamma, alpha, beta);
  isect.t = t;
  isect.n = alpha * tri.n2 + beta * tri.n3 + gamma * tri.n1;
  isect.c = alpha * tri.c2 + beta * tri.c3 + gamma * tri.c1;

  if (dot(isect.n, r.d) > 0.0) isect.n = -isect.n;

  return true;
}

bool traceRay(inout Ray r, out Intersection isect, bool stopIfHit)
{
    isect.t = 10000.0;
    isect.n = vec3(0.0);
    isect.c = vec3(0.0);

    bool hit = false;

    for (int i = 0; i < numTriangles; i++)
    {
        Triangle tri;

        tri.i1 = int(indicies[i * 3    ]);
        tri.i2 = int(indicies[i * 3 + 1]);
        tri.i3 = int(indicies[i * 3 + 2]);

        tri.p1 = vertices[tri.i1 * 3].xyz;
        tri.p2 = vertices[tri.i2 * 3].xyz;
        tri.p3 = vertices[tri.i3 * 3].xyz;

        hit = intersect(r, tri, isect) || hit;

        if (hit && stopIfHit) return true;
    }

    return hit;
}

vec3 reinhard(vec3 v)
{
    return v / (1.0f + v);
}

struct RayStack
{
    vec3 hitPos;
    vec3 nextDir;
    vec3 hitNormal;
    vec3 hitAlbedo;
    vec3 wIn;
};

const int maxDepth = 5;

void main() {
    vec4 projPos = vec4(screenUV, 1.0, 1.0);
    vec4 viewPos = projInvMtx * projPos; viewPos /= viewPos.w;
    vec3 worldDir = normalize((viewInvMtx * viewPos).xyz);

    vec4 camPos = viewInvMtx * vec4(0.0, 0.0, 0.0, 1.0); camPos /= camPos.w;

    Intersection isect;
    Ray r;

    RayStack stack[maxDepth];

    r.o = camPos.xyz;
    r.d = worldDir;
    r.min_t = 0.00001;
    r.max_t = 10000.0;

    float jitter = bayer64(gl_FragCoord.st);

    for (int i = 0; i < maxDepth; i++)
    {
        stack[i].hitPos = vec3(0.0);
        stack[i].nextDir = vec3(0.0);
        stack[i].hitNormal = vec3(0.0);
        stack[i].hitAlbedo = vec3(0.0);
        stack[i].wIn = vec3(0.0);
    }

    for (int depth = 0; depth < maxDepth; depth++)
    {
        if (traceRay(r, isect, false))
        {
            vec3 hitPos = r.max_t * r.d + r.o;

            stack[depth].hitPos = hitPos;
            stack[depth].hitNormal = isect.n;
            stack[depth].hitAlbedo = isect.c;

            // Direct Lighting
            for (int i = 0; i < numLights; i++)
            {
                vec3 lightPos = lights[i].pos.xyz;
                vec3 lightRadiance = lights[i].radiance.rgb;

                vec3 lightDir = normalize(lightPos - hitPos);

                Intersection isectDirectLighting;
                Ray rLight;

                rLight.o = hitPos;
                rLight.d = lightDir;
                rLight.min_t = 0.00001;
                rLight.max_t = distance(lightPos, hitPos) - 0.00001;

                if (!traceRay(rLight, isectDirectLighting, true))
                {
                    vec3 posDiff = lightPos - hitPos;
                    float falloff = 1.0 / (1.0 + dot(posDiff, posDiff)) * max(0.0, dot(isect.n, lightDir));
                    stack[depth].wIn += falloff * isect.c * lightRadiance;
                }
            }

            // Secondary Contribution
            vec2 gridSample = WeylNth(int(jitter * 64 * 64 * 3) + depth);
            stack[depth].nextDir = make_coord_space(isect.n) * cosineHemisphere(gridSample);

            r.o = hitPos;
            r.d = stack[depth].nextDir;
            r.min_t = 0.00001;
            r.max_t = 10000.0;
        }
        else
        {
            break;
        }
    }

    // Resolve
    vec3 L = vec3(0.0);
    for (int depth = maxDepth - 1; depth >= 0; depth--)
    {
        L += stack[depth].wIn;
        L *= stack[depth].hitAlbedo; // hemisphere samples
    }

    outColor = vec4(L, 1.0);
}