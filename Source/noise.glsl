float bayer2(vec2 a)
{
    a = floor(a);
    return fract(dot(a, vec2(0.5f, a.y * 0.75f)));
}

#define bayer4(a)   (bayer2( 0.5f*(a)) * 0.25f + bayer2(a))
#define bayer8(a)   (bayer4( 0.5f*(a)) * 0.25f + bayer2(a))
#define bayer16(a)  (bayer8( 0.5f*(a)) * 0.25f + bayer2(a))
#define bayer32(a)  (bayer16(0.5f*(a)) * 0.25f + bayer2(a))
#define bayer64(a)  (bayer32(0.5f*(a)) * 0.25f + bayer2(a))

float bayer_4x4(in vec2 pos, in vec2 view)
{
	  return bayer4(pos * view);
}

float bayer_8x8(in vec2 pos, in vec2 view)
{
	  return bayer8(pos * view);
}

float bayer_16x16(in vec2 pos, in vec2 view)
{
	  return bayer16(pos * view);
}

float bayer_32x32(in vec2 pos, in vec2 view)
{
	  return bayer32(pos * view);
}

float bayer_64x64(in vec2 pos, in vec2 view)
{
	  return bayer64(pos * view);
}

f16vec2 WeylNth(int n)
{
	  return f16vec2(fract(vec2(n * 12664745, n * 9560333) / exp2(24.0)));
}

f16vec3 cosineHemisphere(f16vec2 i)
{
    float16_t theta = 2.0hf * 3.1415926hf * i.y;
    float16_t sqrtPhi = sqrt(i.x);

    return f16vec3(cos(theta) * sqrtPhi, sin(theta) * sqrtPhi, sqrt(1.0hf - i.x));
}

mat3 make_coord_space(vec3 n)
{
    vec3 h = n;
    if (abs(h.x) <= abs(h.y) && abs(h.x) <= abs(h.z))
        h.x = 1.0;
    else if (abs(h.y) <= abs(h.x) && abs(h.y) <= abs(h.z))
        h.y = 1.0;
    else
        h.z = 1.0;

    vec3 y = normalize(cross(h, n));
    vec3 x = normalize(cross(n, y));

    return mat3(x, y, n);
}

f16vec3 to_coord_space(f16vec3 n, f16vec3 r)
{
    f16vec3 h = n;
    if (abs(h.x) <= abs(h.y) && abs(h.x) <= abs(h.z))
        h.x = 1.0hf;
    else if (abs(h.y) <= abs(h.x) && abs(h.y) <= abs(h.z))
        h.y = 1.0hf;
    else
        h.z = 1.0hf;

    f16vec3 y = normalize(cross(h, n));
    f16vec3 x = normalize(cross(n, y));

    return r.x * x + r.y * y + r.z * n;
}
