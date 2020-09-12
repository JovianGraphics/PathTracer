#pragma once

#include <Ganymede/Source/Ganymede.h>

#include <vector>

#include "ShaderData.h"

class BBox
{
public:
    glm::vec3 a = glm::vec3(0.0);
    glm::vec3 b = glm::vec3(0.0);

    bool empty = false;

    void Extend(BBox other);
    void Extend(glm::vec3 p);
    BBox Intersect(BBox other);
    glm::vec3 GetSize();

    BBox() { empty = true; }
    BBox(glm::vec3 p) : a(p), b(p) {}
    BBox(glm::vec3 a, glm::vec3 b) : a(a), b(b) {}
};

struct BVHNode
{
    glm::vec3 a;
    int32 next;
    glm::vec3 b;
    int32 right;
};

std::vector<BVHNode> BuildBVH(const std::vector<glm::vec4>& vertices, std::vector<uint32>& indices);
void VisualizeBVH(std::vector<BVHNode>& nodes, std::vector<glm::vec4>& vertices, std::vector<VertexAux>& vertexAux, std::vector<glm::uint32>& indices);