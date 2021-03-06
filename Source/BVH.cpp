#include "BVH.h"

#include <queue>

void BBox::Extend(BBox other)
{
    if (other.empty) return;
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

BBox BBox::Intersect(BBox other)
{
    if (empty || other.empty) return BBox();

    glm::vec3 pMax = glm::min(b, other.b);
    glm::vec3 pMin = glm::max(a, other.a);

    glm::vec3 size = pMax - pMin;

    if (size.x < 0.0 || size.y < 0.0 || size.z < 0.0) return BBox((pMax + pMin) * 0.5f);

    return BBox(pMin, pMax);
}

glm::vec3 BBox::GetSize()
{
    return glm::max(glm::vec3(0.0), b - a);
}

struct BuildBVHTask
{
    BBox bbox;
    uint32 start;
    uint32 end;
    uint32 depth;
    bool isRight;
    int32 writeTo;
};

inline BBox ComputeBBox(const std::vector<glm::vec4>& vertices, const std::vector<uint32>& indices, uint32 start, uint32 end)
{
    BBox bbox = BBox();

    for (uint32 i = start; i < end; i += 3)
    {
        glm::vec3 p0 = vertices[indices[i]];
        glm::vec3 p1 = vertices[indices[i + 1]];
        glm::vec3 p2 = vertices[indices[i + 2]];

        bbox.Extend(p0);
        bbox.Extend(p1);
        bbox.Extend(p2);
    }

    return bbox;
}

std::vector<BVHNode> BuildBVH(const std::vector<glm::vec4>& vertices, std::vector<uint32>& indices, float& progress)
{
    std::vector<BVHNode> nodes;
    std::vector<BuildBVHTask> tasks;
    nodes.reserve(indices.size() / 3 * 2);
    tasks.reserve(size_t(log2(indices.size())));
    uint32 processedTriangles = 0;

    tasks.push_back({ ComputeBBox(vertices, indices, 0, indices.size()), 0, uint32(indices.size()), 0, false, 0 });

    uint32 maxDepth = 0;

    while (!tasks.empty())
    {
        BuildBVHTask task = tasks.back();
        tasks.pop_back();

        int32 index = int32(nodes.size());

        if (task.isRight)
        {
            nodes[task.writeTo].right = index;
        }

        nodes.push_back(BVHNode{});
        BVHNode& node = nodes.back();

        if (task.depth > maxDepth) maxDepth = uint32(task.depth);

        uint32 start = task.start;
        uint32 end = task.end;

        BBox bbox = ComputeBBox(vertices, indices, start, end);

        node.a = bbox.a - glm::vec3(0.00006f);
        node.b = bbox.b + glm::vec3(0.00006f);

        if (end - start <= 3)
        {
            if (end - start != 3) throw std::runtime_error("Non-triangle found in primitives");

            node.right = -int32(start);

            processedTriangles++;
            progress = float(processedTriangles) / float(indices.size() / 3);

            continue;
        }

        glm::vec3 selectedAxis = glm::vec3(0.0);
        const float splitRatio[9] = { -1e24, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1e24 };
        glm::vec3 bboxSize = bbox.GetSize();
        float totalArea = dot(bboxSize, bboxSize);
        float maxScore = -(1.0f + 2.0f * (end - start) / 3);
        float selectedRatio = 0.5;
        bool leftFirst = true;
        BBox bboxLeft, bboxRight;

        float radius = glm::distance(bbox.a, bbox.b);
        glm::vec3 bboxCentroid = (bbox.a + bbox.b) * 0.5f;

        const glm::vec3 availableAxes[] = {
            glm::vec3(1.0, 0.0, 0.0),
            glm::vec3(0.0, 1.0, 0.0),
            glm::vec3(0.0, 0.0, 1.0),
            glm::normalize(glm::vec3(-1.0, 0.0, 1.0)),
            glm::normalize(glm::vec3(1.0, 0.0, 1.0)),
            glm::normalize(glm::vec3(1.0, -1.0, 0.0)),
            glm::normalize(glm::vec3(-1.0, -1.0, 0.0)),
            glm::normalize(glm::vec3(0.0, 1.0, -1.0)),
            glm::normalize(glm::vec3(0.0, -1.0, -1.0)),
        };

        for (uint32 x = 0; x < 9; x++)
        {
            glm::vec3 axis = availableAxes[x];

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
                    float splitPos0 = glm::dot(bboxCentroid, axis) + radius * (splitRatio[j] - 0.5);
                    float splitPos1 = glm::dot(bboxCentroid, axis) + radius * (splitRatio[j + 1] - 0.5);

                    float l = dot(centroid, axis);

                    if (l > splitPos0 - 0.00001 && l < splitPos1 + 0.00001)
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
                glm::vec3 leftSize = left.GetSize();
                glm::vec3 rightSize = right.GetSize();
                BBox intersection = left.Intersect(right);

                float pA = dot(leftSize, leftSize) / totalArea;
                float pB = dot(rightSize, rightSize) / totalArea;

                float score = -(
                    1.0f + 
                    pA * leftCount * 2.0f +
                    pB * rightCount * 2.0f
                );

                if (score > maxScore)
                {
                    maxScore = score;
                    selectedRatio = splitRatio[i];
                    selectedAxis = axis;
                    leftFirst = pA < pB;

                    bboxLeft = left;
                    bboxRight = right;
                }
            }
        }

        float splitPos = glm::dot(bboxCentroid, selectedAxis) + radius * (selectedRatio - 0.5);

        std::vector<uint32> leftPrims;
        std::vector<uint32> rightPrims;

        for (uint32 i = start; i < end; i += 3)
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

        uint32 center = start + ((end - start) / 3) / 2 * 3;

        if (leftPrims.size() == 0 || rightPrims.size() == 0)
        {
            tasks.push_back({ ComputeBBox(vertices, indices, center, end), center, end, task.depth + 1, true, index });
            tasks.push_back({ ComputeBBox(vertices, indices, start, center), start, center, task.depth + 1, false, index });
        }
        else
        {
            tasks.push_back({ bboxRight, start + uint32(leftPrims.size()), end, task.depth + 1, true, index });
            tasks.push_back({ bboxLeft, start, start + uint32(leftPrims.size()), task.depth + 1, false, index });
        }
    }

    // Fill in the skip connections (next*)
    struct ConstructNext
    {
        int32 index;
        int32 next;
    };

    std::vector<ConstructNext> indexStack = { { 0, 0 } };
    while (!indexStack.empty())
    {
        ConstructNext c = indexStack.back();
        indexStack.pop_back();

        nodes[c.index].next = c.next;

        // Left tree
        if (nodes[c.index].right > 0)
        {
            indexStack.push_back({ c.index + 1, nodes[c.index].right });
            indexStack.push_back({ nodes[c.index].right, c.next });
        }
    }

    GanymedePrint "Built BVH with", nodes.size(), "nodes (", sizeof(BVHNode) * nodes.size() ,"Bytes),", indices.size() / 3, "triangles, maxDepth =", maxDepth;

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