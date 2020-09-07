#pragma once

#include <Ganymede/Source/Ganymede.h>

#include <vector>

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/type_precision.hpp>

class BBox
{
public:
    glm::vec3 a = glm::vec3(0.0);
    glm::vec3 b = glm::vec3(0.0);

    void Extend(BBox other);
    void Extend(glm::vec3 p);

    BBox(glm::vec3 p) : a(p), b(p) {}
    BBox(glm::vec3 a, glm::vec3 b) : a(a), b(b) {}
};

struct BVHNode
{
    glm::vec3 a;
    uint16 left;
    uint16 right;
    glm::vec3 b;
    uint16 startPrim;
    uint16 endPrim;
};

std::vector<BVHNode> BuildBVH(std::vector<glm::vec4> vertices, std::vector<uint16>& indices);