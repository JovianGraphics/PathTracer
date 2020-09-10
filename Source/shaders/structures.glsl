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

struct Ray
{
    vec3 o;
    float min_t;
    vec3 d;
    float max_t;
    f16vec3 rcpD;
};

struct Triangle
{
    vec3 p1, p2, p3;
    int i1, i2, i3;
};

struct Intersection
{
    f16vec3 bary;
    int i1, i2, i3;
};

struct BVHNode
{
    vec3 a;
    int next;
    vec3 b;
    int right;
    ivec3 index;
};

struct RayStackBuffer
{
	vec3 rayDirection;
    vec3 rayOrigin;
    uint currentDepth;
	f16vec4 hitAlbedo;
    f16vec3 wIn;
    float16_t prob;
};

struct JobDesc
{
    uint32_t index;
};