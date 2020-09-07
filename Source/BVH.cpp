#include "BVH.h"

void BBox::Extend(BBox other)
{
    if (empty)
    {
        a = other.a;
        b = other.b;
        empty = false;
    }
    else
    {
        a = glm::min(a, other.a);
        b = glm::max(b, other.b);
    }
}

void BBox::Extend(glm::vec3 p)
{
    if (empty)
    {
        a = p;
        b = p;
        empty = false;
    }
    else
    {
        a = glm::min(a, p);
        b = glm::max(b, p);
    }
}

uint16 BuildBVH(std::vector<BVHNode>& nodes, std::vector<glm::vec4>& vertices, std::vector<uint16>& indices, uint32 start, uint32 end)
{
    BBox bbox = BBox(vertices[indices[start]]);
    BBox centroidBox = BBox(glm::vec3(100000000.0), glm::vec3(-100000000.0));

    nodes.push_back(BVHNode{});
    
    uint32 index = nodes.size() - 1;

    for (uint32 i = start; i < end; i += 3)
    {
        glm::vec3 p0 = vertices[indices[i]];
        glm::vec3 p1 = vertices[indices[i + 1]];
        glm::vec3 p2 = vertices[indices[i + 2]];

        glm::vec3 centroid = (p0 + p1 + p2) / 3.0f;

        bbox.Extend(p0);
        bbox.Extend(p1);
        bbox.Extend(p2);

        centroidBox.Extend(centroid);
    }

    nodes[index].a = bbox.a - glm::vec3(0.00006);
    nodes[index].b = bbox.b + glm::vec3(0.00006);

    if (end - start <= 3 * 2)
    {
        nodes[index].left = 0;
        nodes[index].right = 0;
        nodes[index].startPrim = start;
        nodes[index].endPrim = end;

        GanymedePrint "BVH Leaf", index, start, end;

        return index;    
    }

    GanymedePrint "BVH Node", index, start, end;

    uint16 selectedAxis = 0; // 0: X, 1: Y, 2: Z
    glm::vec3 axis = glm::vec3(0.0);
    glm::vec3 bboxSize = centroidBox.b - centroidBox.a;
    float maxLength = glm::max(bboxSize.x, glm::max(bboxSize.y, bboxSize.z));

    if (maxLength == bboxSize.x)
    {
        selectedAxis = 0;
        axis = glm::vec3(1.0, 0.0, 0.0);
    }
    else if (maxLength == bboxSize.y)
    {
        selectedAxis = 1;
        axis = glm::vec3(0.0, 1.0, 0.0);
    }
    else if (maxLength == bboxSize.z)
    {
        selectedAxis = 2;
        axis = glm::vec3(0.0, 0.0, 1.0);
    }

    // Binned SAH
    const float splitRatio[9] = { 0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0 };
    BBox bboxBins[8];
    uint32 numPrimBins[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    float totalArea = dot(bbox.b - bbox.a, bbox.b - bbox.a);
    float maxScore = 0.0;
    float selectedRatio = 0.5;

    for (uint32 i = start; i < end; i += 3)
    {
        glm::vec3 p0 = vertices[indices[i]];
        glm::vec3 p1 = vertices[indices[i + 1]];
        glm::vec3 p2 = vertices[indices[i + 2]];

        glm::vec3 centroid = (p0 + p1 + p2) / 3.0f;

        for (uint32 j = 0; j < 8; j++)
        {
            float splitPos0 = glm::dot(centroidBox.a, axis) + maxLength * splitRatio[j];
            float splitPos1 = glm::dot(centroidBox.a, axis) + maxLength * splitRatio[j + 1];

            if (dot(centroid, axis) >= splitPos0 && dot(centroid, axis) <= splitPos1)
            {
                bboxBins[j].Extend(p0);
                bboxBins[j].Extend(p1);
                bboxBins[j].Extend(p2);
                numPrimBins[j]++;
            }
        }
    }

    for (uint32 i = 1; i < 8; i++)
    {
        BBox left; BBox right;
        uint32 leftCount = 0; uint32 rightCount = 0;
        for (uint32 j = 0; j < i; j++)
        {
            left.Extend(bboxBins[j]);
            leftCount += numPrimBins[j];
        }
        for (uint32 j = i; j < 8; j++)
        {
            right.Extend(bboxBins[j]);
            rightCount += numPrimBins[j];
        }
        // From cost to score ...
        float score = glm::exp2(-(1.5 + (dot(left.b - left.a, left.b - left.a) * leftCount + dot(right.b - right.a, right.b - right.a) * rightCount) * 2.0 / totalArea));

        if (score > maxScore)
        {
            maxScore = score;
            selectedRatio = splitRatio[i];
        }
    }

    float splitPos = glm::dot(centroidBox.a, axis) + maxLength * selectedRatio;

    std::vector<uint16> leftPrims;
    std::vector<uint16> rightPrims;

    for (int i = start; i < end; i += 3)
    {
        glm::vec3 p0 = vertices[indices[i]];
        glm::vec3 p1 = vertices[indices[i + 1]];
        glm::vec3 p2 = vertices[indices[i + 2]];

        glm::vec3 centroid = (p0 + p1 + p2) / 3.0f;

        if (dot(centroid, axis) > splitPos)
        {
            rightPrims.push_back(indices[i]);
            rightPrims.push_back(indices[i + 1]);
            rightPrims.push_back(indices[i + 2]);
        }
        else
        {
            leftPrims.push_back(indices[i]);
            leftPrims.push_back(indices[i + 1]);
            leftPrims.push_back(indices[i + 2]);
        }
    }

    int i = start;
    for (int j = 0; j < leftPrims.size(); j++)
    {
        indices[i] = leftPrims[j];
        i++;
    }
    for (int j = 0; j < rightPrims.size(); j++)
    {
        indices[i] = rightPrims[j];
        i++;
    }

    nodes[index].startPrim = 0;
    nodes[index].endPrim = 0;

    uint16 center = start + ((end - start) / 3) / 2 * 3;

    if (leftPrims.size() == 0)
    {
        nodes[index].left = BuildBVH(nodes, vertices, indices, start, center);
        nodes[index].right = BuildBVH(nodes, vertices, indices, center, end);
    }
    else if (rightPrims.size() == 0)
    {
        nodes[index].left = BuildBVH(nodes, vertices, indices, start, center);
        nodes[index].right = BuildBVH(nodes, vertices, indices, center, end);
    }
    else
    {
        nodes[index].left = BuildBVH(nodes, vertices, indices, start, start + leftPrims.size());
        nodes[index].right = BuildBVH(nodes, vertices, indices, start + leftPrims.size(), end);
    }

    GanymedePrint "Node", index, "L", nodes[index].left, "R", nodes[index].right;

    return index;
}

std::vector<BVHNode> BuildBVH(std::vector<glm::vec4> vertices, std::vector<uint16>& indices)
{
    std::vector<BVHNode> nodes;

    BuildBVH(nodes, vertices, indices, 0, indices.size());

    return nodes;
}

void VisualizeBVH(std::vector<BVHNode>& nodes, std::vector<glm::vec4>& vertices, std::vector<VertexAux>& vertexAux, std::vector<glm::uint16>& indices)
{
    uint16 startVertex = 0;
    for (uint32 i = 0; i < nodes.size(); i++)
    {
        glm::vec3 a = nodes[i].a;
        glm::vec3 b = nodes[i].b;

        vertices.push_back(glm::vec4(a.x, a.y, a.z, 1.0));
        vertices.push_back(glm::vec4(a.x, a.y, b.z, 1.0));
        vertices.push_back(glm::vec4(a.x, b.y, a.z, 1.0));
        vertices.push_back(glm::vec4(a.x, b.y, b.z, 1.0));
        vertices.push_back(glm::vec4(b.x, a.y, a.z, 1.0));
        vertices.push_back(glm::vec4(b.x, a.y, b.z, 1.0));
        vertices.push_back(glm::vec4(b.x, b.y, a.z, 1.0));
        vertices.push_back(glm::vec4(b.x, b.y, b.z, 1.0));

        vertexAux.push_back(VertexAux{ glm::vec3(0.0), glm::u8vec4(127) });
        vertexAux.push_back(VertexAux{ glm::vec3(0.0), glm::u8vec4(127) });
        vertexAux.push_back(VertexAux{ glm::vec3(0.0), glm::u8vec4(127) });
        vertexAux.push_back(VertexAux{ glm::vec3(0.0), glm::u8vec4(127) });
        vertexAux.push_back(VertexAux{ glm::vec3(0.0), glm::u8vec4(127) });
        vertexAux.push_back(VertexAux{ glm::vec3(0.0), glm::u8vec4(127) });
        vertexAux.push_back(VertexAux{ glm::vec3(0.0), glm::u8vec4(127) });
        vertexAux.push_back(VertexAux{ glm::vec3(0.0), glm::u8vec4(127) });

        indices.push_back(startVertex + 0);
        indices.push_back(startVertex + 1);
        indices.push_back(startVertex + 4);
        indices.push_back(startVertex + 5);
        indices.push_back(startVertex + 0);
        indices.push_back(startVertex + 4);
        indices.push_back(startVertex + 1);
        indices.push_back(startVertex + 5);

        indices.push_back(startVertex + 2);
        indices.push_back(startVertex + 3);
        indices.push_back(startVertex + 6);
        indices.push_back(startVertex + 7);
        indices.push_back(startVertex + 2);
        indices.push_back(startVertex + 6);
        indices.push_back(startVertex + 3);
        indices.push_back(startVertex + 7);

        indices.push_back(startVertex + 0);
        indices.push_back(startVertex + 2);
        indices.push_back(startVertex + 1);
        indices.push_back(startVertex + 3);
        indices.push_back(startVertex + 4);
        indices.push_back(startVertex + 6);
        indices.push_back(startVertex + 5);
        indices.push_back(startVertex + 7);

        startVertex += 8;
    }
}