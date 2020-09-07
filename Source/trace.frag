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

struct Light
{
    vec3 pos;
    vec3 radiance;
};

struct VertexAux
{
    vec3 normal;
    u8vec4 color;
};

struct BVHNode
{
    vec3 a;
    uint16_t left;
    uint16_t right;
    vec3 b;
    uint16_t startPrim;
    uint16_t endPrim;
};

layout(std430, binding = 1) buffer lightBuffer
{
    Light lights[];
};

layout(std430, binding = 2) buffer vertexBufferAux
{
    VertexAux vertexAux[];
};

layout(std430, binding = 3) buffer indexBuffer
{
    layout(align = 8) uint16_t indicies[];
};

layout(std430, binding = 4) buffer blueNoiseBuffer
{
    uint16_t blueNoise[];
};

layout(binding = 5, rgba32f) uniform image2D prevAccumulation;
layout(binding = 6, rgba32f) uniform image2D currAccumulation;

layout(std430, binding = 7) buffer vertexBufferPos
{
    vec4 vertices[];
};

layout(std430, binding = 8) buffer bvhBuffer
{
    BVHNode bvh[];
};

layout(location = 0) in vec2 screenUV;

layout(location = 0) out vec4 outColor;

#include "noise.glsl"

struct Ray
{
  vec3 o;
  vec3 d;
  vec3 rcpD;
  float min_t;
  float max_t;
};

struct Triangle
{
  int i1, i2, i3;
  vec3 p1, p2, p3;
  f16vec3 n1, n2, n3;
  f16vec4 c1, c2, c3;
};

struct Intersection
{
  f16vec3 bary;
  f16vec3 n;
  f16vec4 c;
  float t;
};

bool intersectBBox(Ray r, vec3 a, vec3 b)
{
    float _t0, _t1;

    vec3 rrd = r.rcpD;

    _t0 = (a.x - r.o.x) * rrd.x;
    _t1 = (b.x - r.o.x) * rrd.x;

    float tmin = min(_t0, _t1);
    float tmax = max(_t0, _t1);

    _t0 = (a.y - r.o.y) * rrd.y;
    _t1 = (b.y - r.o.y) * rrd.y;

    tmin = max(tmin, min(_t0, _t1));
    tmax = min(tmax, max(_t0, _t1));

    _t0 = (a.z - r.o.z) * rrd.z;
    _t1 = (b.z - r.o.z) * rrd.z;

    tmin = max(tmin, min(_t0, _t1));
    tmax = min(tmax, max(_t0, _t1));

    if (tmax >= tmin) {
        if (r.min_t < tmax && r.max_t > tmin)
        {
            return true;
        }
    }

    return false;
}

bool intersect(inout Ray r, Triangle tri, inout Intersection isect, bool stopIfHit)
{
  vec3 e1 = tri.p2 - tri.p1;
  vec3 e2 = tri.p3 - tri.p1;
  vec3 s = r.o - tri.p1, s1 = cross(r.d, e2), s2 = cross(s, e1);
  vec3 matrix = vec3(dot(s2, e2), dot(s1, s), dot(s2, r.d));
  vec3 intersection = matrix / dot(s1, e1);

  float t = intersection.x;
  float16_t alpha = float16_t(intersection.y);
  float16_t beta = float16_t(intersection.z);
  float16_t gamma = 1.0hf - alpha - beta;

  if (t < r.min_t || t > r.max_t || alpha < 0.0 || beta < 0.0 || gamma < 0.0) return false;

  r.max_t = t;

  if (!stopIfHit)
  {
      tri.c1 = f16vec4(vertexAux[tri.i1].color) * (1.0hf / 255.0hf);
      tri.c2 = f16vec4(vertexAux[tri.i2].color) * (1.0hf / 255.0hf);
      tri.c3 = f16vec4(vertexAux[tri.i3].color) * (1.0hf / 255.0hf);

      tri.n1 = f16vec3(vertexAux[tri.i1].normal).xyz;
      tri.n2 = f16vec3(vertexAux[tri.i2].normal).xyz;
      tri.n3 = f16vec3(vertexAux[tri.i3].normal).xyz;

      isect.bary = f16vec3(gamma, alpha, beta);
      isect.t = t;
      isect.n = alpha * tri.n2 + beta * tri.n3 + gamma * tri.n1;
      isect.c = alpha * tri.c2 + beta * tri.c3 + gamma * tri.c1;

      if (dot(isect.n, f16vec3(r.d)) > 0.0) isect.n = -isect.n;
  }

  return true;
}

bool traceRayTriangles(inout Ray r, int start, int end, out Intersection isect, bool stopIfHit)
{
    bool hit = false;

    for (int i = start; i < end; i += 3)
    {
        Triangle tri;

        tri.i1 = int(indicies[i   ]);
        tri.i2 = int(indicies[i + 1]);
        tri.i3 = int(indicies[i + 2]);

        tri.p1 = vertices[tri.i1].xyz;
        tri.p2 = vertices[tri.i2].xyz;
        tri.p3 = vertices[tri.i3].xyz;

        hit = intersect(r, tri, isect, stopIfHit) || hit;

        if (hit && stopIfHit) return true;
    }

    return hit;
}

bool traceRay(inout Ray r, out Intersection isect, bool stopIfHit)
{
    isect.t = 10000.0;
    isect.n = f16vec3(0.0);
    isect.c = f16vec4(1.0);

    bool hit = false;

    uint stack[64];
    stack[0] = 0;
    uint stackSize = 1;

    while (stackSize > 0)
    {
        uint index = stack[stackSize - 1];
        stackSize -= 1;
        
        if (bvh[index].left != bvh[index].right)
        {
            // Not a leaf
            uint left = uint(bvh[index].left);
            uint right = uint(bvh[index].right);

            if (intersectBBox(r, bvh[left].a, bvh[left].b))
            {
                stack[stackSize] = left;
                stackSize++;
            }

            if (intersectBBox(r, bvh[right].a, bvh[right].b))
            {
                stack[stackSize] = right;
                stackSize++;
            }
        }
        else
        {
            // Leaf node
            hit = traceRayTriangles(r, bvh[index].startPrim, bvh[index].endPrim, isect, stopIfHit) || hit;
            if (hit && stopIfHit) return true;
        }
    }

    return hit;

    while (stackSize > 0)
    {
        uint index = stack[stackSize - 1];
        stackSize -= 1;

        if (bvh[index].left != bvh[index].right)
        {
            // Not a leaf
            stack[stackSize] = uint(bvh[index].left);
            stack[stackSize + 1] = uint(bvh[index].right);
            stackSize += 2;
        }
        else
        {
            // Leaf node
            hit = traceRayTriangles(r, bvh[index].startPrim, bvh[index].endPrim, isect, stopIfHit) || hit;
        }

        if (hit && stopIfHit) return true;
    }

    return hit;
}

f16vec3 reinhard(f16vec3 v)
{
    return v / (1.0hf + v);
}

struct RayStack
{
    vec3 hitPos;
    vec3 nextDir;
    f16vec3 hitNormal;
    f16vec4 hitAlbedo;
    f16vec3 wIn;
};

const int maxDepth = 5;

void main() {
    int jitter = (blueNoise[(uint(gl_FragCoord.x) & 0xFF) + ((uint(gl_FragCoord.y) & 0xFF) << 8)] * 256 + blueNoise[frameIndex]) & 0xFFFF;

    vec4 projPos = vec4(screenUV + (WeylNth(jitter) * 2.0 - 1.0) / viewportSize, 1.0, 1.0);
    vec4 viewPos = projInvMtx * projPos; viewPos /= viewPos.w;
    vec3 worldDir = normalize((viewInvMtx * viewPos).xyz);

    vec4 camPos = viewInvMtx * vec4(0.0, 0.0, 0.0, 1.0); camPos /= camPos.w;

    vec3 accumulation = vec3(0.0);

    for (uint n = 0; n < numRays; n++)
    {
        Intersection isect;
        Ray r;

        RayStack stack[maxDepth];

        r.o = camPos.xyz;
        r.d = worldDir;
        r.rcpD = 1.0 / r.d;
        r.min_t = 0.00005;
        r.max_t = 10000.0;

        for (int i = 0; i < maxDepth; i++)
        {
            stack[i].hitPos = vec3(0.0);
            stack[i].nextDir = vec3(0.0);
            stack[i].hitNormal = f16vec3(0.0);
            stack[i].hitAlbedo = f16vec4(0.0);
            stack[i].wIn = f16vec3(0.0);
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
                if (isect.c.a > 0.5)
                {
                    for (int i = 0; i < numLights; i++)
                    {
                        vec3 lightPos = lights[i].pos.xyz;

                        vec3 lightDir = normalize(lightPos - hitPos);

                        Intersection isectDirectLighting;
                        Ray rLight;

                        rLight.o = hitPos;
                        rLight.d = lightDir;
                        rLight.rcpD = 1.0 / rLight.d;
                        rLight.min_t = 0.00005;
                        rLight.max_t = distance(lightPos, hitPos) - 0.00005;

                        if (!traceRay(rLight, isectDirectLighting, true))
                        {
                            f16vec3 lightRadiance = f16vec3(lights[i].radiance.rgb);
    
                            vec3 posDiff = lightPos - hitPos;
                            float16_t falloff = 1.0hf / (1.0hf + float16_t(dot(posDiff, posDiff))) * max(0.0hf, dot(isect.n, f16vec3(lightDir)));
                            stack[depth].wIn += falloff * isect.c.rgb * lightRadiance;
                        }
                    }
                }

                // Secondary Contribution
                f16vec2 gridSample = WeylNth(int(jitter * numRays + n + depth));
                vec3 nextDir = vec3(to_coord_space(isect.n, cosineHemisphere(gridSample)));

                if (isect.c.a < 0.5)
                {
                    float16_t ior = 1.3hf;
                    if (depth > 0 && stack[depth - 1].hitAlbedo.a < 0.5) ior = 1.0hf / 1.3hf;

                    nextDir = vec3(refract(f16vec3(r.d), isect.n, ior));

                    if (nextDir == vec3(0.0)) nextDir = vec3(reflect(f16vec3(r.d), isect.n));
                }
                else if (depth > 0)
                {
                    if (float(jitter) / 65536.0 > 0.7) break;

                    stack[depth].hitAlbedo /= 0.7hf;
                }

                stack[depth].nextDir = nextDir;

                r.o = hitPos;
                r.d = nextDir;
                r.rcpD = 1.0 / r.d;
                r.min_t = 0.00005;
                r.max_t = 10000.0;
            }
            else
            {
                break;
            }
        }

        // Resolve
        f16vec3 L = f16vec3(0.0);
        for (int depth = maxDepth - 1; depth >= 0; depth--)
        {
            L += stack[depth].wIn;
            L *= stack[depth].hitAlbedo.rgb; // hemisphere samples
        }

        accumulation += vec3(L);
    }

    accumulation /= float(numRays);

    vec4 prev = imageLoad(prevAccumulation, ivec2(gl_FragCoord.st));
    prev += vec4(accumulation, 1.0);
    imageStore(currAccumulation, ivec2(gl_FragCoord.st), prev);

    outColor = vec4(prev.rgb / prev.w, 1.0);
}