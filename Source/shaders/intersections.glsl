bool intersectBBox(Ray r, vec3 _a, vec3 _b)
{
    vec3 c = (_a + _b) * 0.5f;
    f16vec3 a = f16vec3(_a - c);
    f16vec3 b = f16vec3(_b - c);
    f16vec3 o = f16vec3(r.o - c);
    f16vec3 rrd = r.rcpD;

    float16_t _t0, _t1;
   
    _t0 = (a.x - o.x) * rrd.x;
    _t1 = (b.x - o.x) * rrd.x;

    float16_t tmin = min(_t0, _t1);
    float16_t tmax = max(_t0, _t1);

    _t0 = (a.y - o.y) * rrd.y;
    _t1 = (b.y - o.y) * rrd.y;

    tmin = max(tmin, min(_t0, _t1));
    tmax = min(tmax, max(_t0, _t1));

    _t0 = (a.z - o.z) * rrd.z;
    _t1 = (b.z - o.z) * rrd.z;

    tmin = max(tmin, min(_t0, _t1));
    tmax = min(tmax, max(_t0, _t1));

    return (tmax > 0.0hf || tmin > 0.0hf) && tmax >= tmin && r.min_t < tmax && r.max_t > tmin;
}

bool intersect(inout Ray r, Triangle tri, inout Intersection isect, bool stopIfHit)
{
    vec3 e1 = tri.p2 - tri.p1;
    vec3 e2 = tri.p3 - tri.p1;
    vec3 s = r.o - tri.p1, s1 = cross(r.d, e2), s2 = cross(s, e1);
    vec3 matrix = vec3(dot(s2, e2), dot(s1, s), dot(s2, r.d));
    vec3 intersection = matrix / dot(s1, e1);

    float t = intersection.x;
    float alpha = intersection.y;
    float beta = intersection.z;
    float gamma = 1.0f - alpha - beta;

    if (t < r.min_t || t > r.max_t || alpha < 0.0 || beta < 0.0 || gamma < 0.0) return false;

    r.max_t = t;

    isect.bary = f16vec3(gamma, alpha, beta);
    isect.i1 = tri.i1;
    isect.i2 = tri.i2;
    isect.i3 = tri.i3;

    return true;
}

// NOT USED
bool traceRayTriangles(inout Ray r, uint start, uint end, out Intersection isect, bool stopIfHit)
{
    bool hit = false;

    for (uint i = start; i < end; i += 3)
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
    bool hit = false;

    uint index = 0;

    while (index < numBVHNodes)
    {
        BVHNode node = bvh[index];

        if (intersectBBox(r, node.a, node.b))
        {
            if (node.right <= 0)
            {
                // Leaf node
                Triangle tri;

                tri.i1 = node.index.x;
                tri.i2 = node.index.y;
                tri.i3 = node.index.z;

                tri.p1 = vertices[tri.i1].xyz;
                tri.p2 = vertices[tri.i2].xyz;
                tri.p3 = vertices[tri.i3].xyz;

                hit = intersect(r, tri, isect, stopIfHit) || hit;

                if (hit && stopIfHit) return true;

                node = bvh[node.next];
            }

            index++;
        }
        else
        {
            if (node.next == 0) return hit;
            index = node.next;
        }
    }

    return hit;
}