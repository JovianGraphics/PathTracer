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

struct BuildBVHTask
{
    std::vector<glm::vec4>& vertices;
    std::vector<uint32>& indices;
    uint32 start;
    uint32 end;
    size_t resultOffset;
};

uint32 BuildBVH(std::vector<BVHNode>& nodes, std::vector<glm::vec4>& _vertices, std::vector<uint32>& _indices, uint32 _start, uint32 _end)
{
    std::vector<BuildBVHTask> tasks;

    uint32 result;
    tasks.push_back({ _vertices, _indices, _start, _end, offsetof(BVHNode, BVHNode::endPrim) });

    uint32 maxDepth = 0;

    while (!tasks.empty())
    {
        if (tasks.size() > maxDepth) maxDepth = uint32(tasks.size());

        BuildBVHTask& task = tasks.back();
        tasks.pop_back();

        std::vector<glm::vec4>& vertices = task.vertices;
        std::vector<uint32>& indices = task.indices;
        uint32 start = task.start;
        uint32 end = task.end;

        BBox bbox = BBox(vertices[indices[start]]);

        nodes.push_back(BVHNode{});

        uint32 index = nodes.size() - 1;

        *(reinterpret_cast<uint32*>(reinterpret_cast<uint8*>(nodes.data()) + task.resultOffset)) = index;

        for (uint32 i = start; i < end; i += 3)
        {
            glm::vec3 p0 = vertices[indices[i]];
            glm::vec3 p1 = vertices[indices[i + 1]];
            glm::vec3 p2 = vertices[indices[i + 2]];

            bbox.Extend(p0);
            bbox.Extend(p1);
            bbox.Extend(p2);
        }

        nodes[index].a = bbox.a - glm::vec3(0.00006f);
        nodes[index].b = bbox.b + glm::vec3(0.00006f);

        if (end - start <= 3 * 2)
        {
            nodes[index].left = 0;
            nodes[index].right = 0;
            nodes[index].startPrim = start;
            nodes[index].endPrim = end;

            continue;
        }

        glm::vec3 selectedAxis = glm::vec3(0.0);
        const float splitRatio[9] = { 0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0 };
        glm::vec3 bboxSize = bbox.b - bbox.a;
        float totalArea = dot(bbox.b - bbox.a, bbox.b - bbox.a);
        float maxScore = 0.0;
        float selectedRatio = 0.5;

        for (uint32 x = 0; x < 3; x++)
        {
            glm::vec3 axis = glm::vec3(0.0);

            if (x == 0)
                axis = glm::vec3(1.0, 0.0, 0.0);
            else if (x == 1)
                axis = glm::vec3(0.0, 1.0, 0.0);
            else if (x == 2)
                axis = glm::vec3(0.0, 0.0, 1.0);

            float length = dot(bboxSize, axis);

            // Binned SAH
            BBox bboxBins[8];
            uint32 numPrimBins[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };

            for (uint32 i = start; i < end; i += 3)
            {
                glm::vec3 p0 = vertices[indices[i]];
                glm::vec3 p1 = vertices[indices[i + 1]];
                glm::vec3 p2 = vertices[indices[i + 2]];

                glm::vec3 centroid = (p0 + p1 + p2) / 3.0f;

                for (uint32 j = 0; j < 8; j++)
                {
                    float splitPos0 = glm::dot(bbox.a, axis) + length * splitRatio[j];
                    float splitPos1 = glm::dot(bbox.a, axis) + length * splitRatio[j + 1];

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
                float score = glm::exp2(-(1.0 + (dot(left.b - left.a, left.b - left.a) * leftCount + dot(right.b - right.a, right.b - right.a) * rightCount) * 2.0 / totalArea));

                if (score > maxScore)
                {
                    maxScore = score;
                    selectedRatio = splitRatio[i];
                    selectedAxis = axis;
                }
            }
        }

        float splitPos = glm::dot(bbox.a, selectedAxis) + glm::dot(bboxSize, selectedAxis) * selectedRatio;

        std::vector<uint32> leftPrims;
        std::vector<uint32> rightPrims;

        for (int i = start; i < end; i += 3)
        {
            glm::vec3 p0 = vertices[indices[i]];
            glm::vec3 p1 = vertices[indices[i + 1]];
            glm::vec3 p2 = vertices[indices[i + 2]];

            glm::vec3 centroid = (p0 + p1 + p2) / 3.0f;

            if (glm::dot(centroid, selectedAxis) > splitPos)
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

        uint32 center = start + ((end - start) / 3) / 2 * 3;

        if (leftPrims.size() == 0 || rightPrims.size() == 0)
        {
            tasks.push_back({ vertices, indices, start, center, offsetof(BVHNode, BVHNode::left) + index * sizeof(BVHNode) });
            tasks.push_back({ vertices, indices, center, end, offsetof(BVHNode, BVHNode::right) + index * sizeof(BVHNode) });
        }
        else
        {
            tasks.push_back({ vertices, indices, start, start + uint32(leftPrims.size()), offsetof(BVHNode, BVHNode::left) + index * sizeof(BVHNode) });
            tasks.push_back({ vertices, indices, start + uint32(leftPrims.size()), end, offsetof(BVHNode, BVHNode::right) + index * sizeof(BVHNode) });
        }

    }

    GanymedePrint "Built BVH with", nodes.size(), "nodes, maxDepth =", maxDepth;

    return result;
}

std::vector<BVHNode> BuildBVH(std::vector<glm::vec4> vertices, std::vector<uint32>& indices)
{
    std::vector<BVHNode> nodes;

    BuildBVH(nodes, vertices, indices, 0, indices.size());

    return nodes;
}

void VisualizeBVH(std::vector<BVHNode>& nodes, std::vector<glm::vec4>& vertices, std::vector<VertexAux>& vertexAux, std::vector<glm::uint32>& indices)
{
    uint32 startVertex = 0;
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