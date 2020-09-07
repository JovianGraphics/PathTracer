#include "Io/Source/Io.h"
#include "Io/Source/IoEntryFunc.h"
#include "Europa/Source/EuropaVk.h"
#include "Amalthea/Source/Amalthea.h"
#include "Ganymede/Source/Ganymede.h"
#include "Ganymede/Source/GanymedeECS.h"
#include "Himalia/Source/Himalia.h"

#include "trace.frag.h"
#include "trace.vert.h"

#include <thread>
#include <chrono>

#include <glm/gtx/transform.hpp>

struct Vertex
{
	alignas(16) glm::vec3 pos;
	alignas(16) glm::vec3 color;
	alignas(16) glm::vec3 normal;
};

struct ShaderConstants {
	glm::mat4 viewMtx;
	glm::mat4 projMtx;
	glm::mat4 viewInvMtx;
	glm::mat4 projInvMtx;
	uint32 numLights;
	uint32 numTriangles;
};

struct Light
{
	alignas(16) glm::vec3 pos;
	alignas(16) glm::vec3 radiance;
};

std::vector<Vertex> vertices;
std::vector<uint16> indices;
std::vector<Light> lights = {
	{ glm::vec3(0.0, 0.7, 0.0), glm::vec3(1.0, 1.0, 1.0) },
};

class TestApp
{
public:
	Amalthea m_amalthea;

	EuropaBuffer::Ref m_vertexBuffer;
	EuropaBuffer::Ref m_indexBuffer;
	EuropaBuffer::Ref m_lightsBuffer;

	EuropaImage::Ref m_depthImage;
	EuropaImageView::Ref m_depthView;

	EuropaRenderPass::Ref m_mainRenderPass;

	EuropaDescriptorPool::Ref m_descPool;
	EuropaGraphicsPipeline::Ref m_pipeline;
	EuropaPipelineLayout::Ref m_pipelineLayout;

	std::vector<EuropaDescriptorSet::Ref> m_descSets;
	std::vector<EuropaFramebuffer::Ref> m_frameBuffers;

	uint32 m_constantsSize;

	float m_orbitHeight = 1.5;
	float m_orbitRadius = 3.0;
	float m_orbitAngle = 0.0;

	GanymedeScrollingBuffer m_frameTimeLog = GanymedeScrollingBuffer(1000, 0);
	GanymedeScrollingBuffer m_frameRateLog = GanymedeScrollingBuffer(1000, 0);
	uint32 m_frameCount = 0;
	float m_fps = 0.0;

	AmaltheaBehaviors::OnCreateDevice f_onCreateDevice = [&](Amalthea* amalthea)
	{
		// Load Model
		HimaliaPlyModel plyModel;
		plyModel.LoadFile("Assets/cornellBox.ply");

		HimaliaVertexProperty vertexFormat[] = {
			HimaliaVertexProperty::Position,
			HimaliaVertexProperty::Color,
			HimaliaVertexProperty::Normal
		};
		uint32 alignments[] = {
			0, 16, 32
		};
		plyModel.mesh.BuildVertices<Vertex>(vertices, 3, vertexFormat, alignments);
		plyModel.mesh.BuildIndices<uint16>(indices);

		// Create & Upload geometry buffers
		EuropaBufferInfo vertexBufferInfo;
		vertexBufferInfo.exclusive = true;
		vertexBufferInfo.size = uint32(vertices.size() * sizeof(Vertex));
		vertexBufferInfo.usage = EuropaBufferUsage(EuropaBufferUsageStorage | EuropaBufferUsageTransferDst);
		vertexBufferInfo.memoryUsage = EuropaMemoryUsage::GpuOnly;
		m_vertexBuffer = amalthea->m_device->CreateBuffer(vertexBufferInfo);

		amalthea->m_transferUtil->UploadToBufferEx(m_vertexBuffer, vertices.data(), uint32(vertices.size()));
		
		EuropaBufferInfo indexBufferInfo;
		indexBufferInfo.exclusive = true;
		indexBufferInfo.size = uint32(indices.size() * sizeof(uint16));
		indexBufferInfo.usage = EuropaBufferUsage(EuropaBufferUsageStorage | EuropaBufferUsageTransferDst);
		indexBufferInfo.memoryUsage = EuropaMemoryUsage::GpuOnly;
		m_indexBuffer = amalthea->m_device->CreateBuffer(indexBufferInfo);

		amalthea->m_transferUtil->UploadToBufferEx(m_indexBuffer, indices.data(), uint32(indices.size()));

		EuropaBufferInfo lightBufferInfo;
		lightBufferInfo.exclusive = true;
		lightBufferInfo.size = uint32(lights.size() * sizeof(Light));
		lightBufferInfo.usage = EuropaBufferUsage(EuropaBufferUsageStorage | EuropaBufferUsageTransferDst);
		lightBufferInfo.memoryUsage = EuropaMemoryUsage::GpuOnly;
		m_lightsBuffer = amalthea->m_device->CreateBuffer(lightBufferInfo);

		amalthea->m_transferUtil->UploadToBufferEx(m_lightsBuffer, lights.data(), uint32(lights.size()));

		amalthea->m_ioSurface->SetKeyCallback([](uint8 keyAscii, uint16 keyV, std::string, IoKeyboardEvent ev)
			{
				GanymedePrint "Key", keyAscii, keyV, IoKeyboardEventToString(ev);
			});
	};

	AmaltheaBehaviors::OnDestroyDevice f_onDestroyDevice = [&](Amalthea* amalthea)
	{
	};

	AmaltheaBehaviors::OnDestroyDevice f_onCreateSwapChain = [&](Amalthea* amalthea)
	{
		// Create Depth buffer
		EuropaImageInfo depthInfo;
		depthInfo.width = amalthea->m_windowSize.x;
		depthInfo.height = amalthea->m_windowSize.y;
		depthInfo.initialLayout = EuropaImageLayout::Undefined;
		depthInfo.type = EuropaImageType::Image2D;
		depthInfo.format = EuropaImageFormat::D16Unorm;
		depthInfo.usage = EuropaImageUsageDepthStencilAttachment;
		depthInfo.memoryUsage = EuropaMemoryUsage::GpuOnly;

		m_depthImage = amalthea->m_device->CreateImage(depthInfo);

		EuropaImageViewCreateInfo depthViewInfo;
		depthViewInfo.format = EuropaImageFormat::D16Unorm;
		depthViewInfo.image = m_depthImage;
		depthViewInfo.type = EuropaImageViewType::View2D;
		depthViewInfo.minArrayLayer = 0;
		depthViewInfo.minMipLevel = 0;
		depthViewInfo.numArrayLayers = 1;
		depthViewInfo.numMipLevels = 1;

		m_depthView = amalthea->m_device->CreateImageView(depthViewInfo);

		// Create Renderpass
		m_mainRenderPass = amalthea->m_device->CreateRenderPassBuilder();
		uint32 presentTarget = m_mainRenderPass->AddAttachment(EuropaAttachmentInfo{
			EuropaImageFormat::BGRA8sRGB,
			EuropaAttachmentLoadOp::Clear,
			EuropaAttachmentStoreOp::Store,
			EuropaAttachmentLoadOp::DontCare,
			EuropaAttachmentStoreOp::DontCare,
			EuropaImageLayout::Undefined,
			EuropaImageLayout::Present
			});
		uint32 depthTarget = m_mainRenderPass->AddAttachment(EuropaAttachmentInfo{
			EuropaImageFormat::D16Unorm,
			EuropaAttachmentLoadOp::Clear,
			EuropaAttachmentStoreOp::Store,
			EuropaAttachmentLoadOp::DontCare,
			EuropaAttachmentStoreOp::DontCare,
			EuropaImageLayout::Undefined,
			EuropaImageLayout::DepthStencilAttachment
			});
		std::vector<EuropaAttachmentReference> attachments = { { presentTarget, EuropaImageLayout::ColorAttachment } };
		EuropaAttachmentReference depthAttachment = { depthTarget, EuropaImageLayout::DepthStencilAttachment };
		uint32 forwardPass = m_mainRenderPass->AddSubpass(EuropaPipelineBindPoint::Graphics, attachments, &depthAttachment);
		m_mainRenderPass->AddDependency(EuropaRenderPass::SubpassExternal, forwardPass, EuropaPipelineStageColorAttachmentOutput, EuropaAccessNone, EuropaPipelineStageColorAttachmentOutput, EuropaAccessColorAttachmentWrite);
		m_mainRenderPass->CreateRenderpass();

		// Create Pipeline
		EuropaShaderModule::Ref shaderFragment = amalthea->m_device->CreateShaderModule(shader_spv_trace_frag, sizeof(shader_spv_trace_frag));
		EuropaShaderModule::Ref shaderVertex = amalthea->m_device->CreateShaderModule(shader_spv_trace_vert, sizeof(shader_spv_trace_vert));

		EuropaDescriptorSetLayout::Ref descLayout = amalthea->m_device->CreateDescriptorSetLayout();
		descLayout->DynamicUniformBuffer(0, 1, EuropaShaderStageAllGraphics);
		descLayout->Storage(1, 1, EuropaShaderStageAllGraphics);
		descLayout->Storage(2, 1, EuropaShaderStageAllGraphics);
		descLayout->Storage(3, 1, EuropaShaderStageAllGraphics);
		descLayout->Build();

		m_pipelineLayout = amalthea->m_device->CreatePipelineLayout(EuropaPipelineLayoutInfo{ 1, 0, &descLayout });

		EuropaGraphicsPipelineCreateInfo pipelineDesc{};

		EuropaShaderStageInfo stages[2] = {
			EuropaShaderStageInfo{ EuropaShaderStageFragment, shaderFragment, "main" },
			EuropaShaderStageInfo{ EuropaShaderStageVertex, shaderVertex, "main"}
		};

		pipelineDesc.shaderStageCount = 2;
		pipelineDesc.stages = stages;
		pipelineDesc.vertexInput.vertexBindingCount = 0;
		pipelineDesc.vertexInput.vertexBindings = nullptr;
		pipelineDesc.vertexInput.attributeBindingCount = 0;
		pipelineDesc.vertexInput.attributeBindings = nullptr;
		pipelineDesc.viewport.position = glm::vec2(0.0);
		pipelineDesc.viewport.size = amalthea->m_windowSize;
		pipelineDesc.viewport.minDepth = 0.0f;
		pipelineDesc.viewport.maxDepth = 1.0f;
		pipelineDesc.rasterizer.cullBackFace = false;
		pipelineDesc.scissor.position = glm::vec2(0.0);
		pipelineDesc.scissor.size = amalthea->m_windowSize;
		pipelineDesc.depthStencil.enableDepthTest = true;
		pipelineDesc.depthStencil.enableDepthWrite = true;
		pipelineDesc.layout = m_pipelineLayout;
		pipelineDesc.renderpass = m_mainRenderPass;
		pipelineDesc.targetSubpass = 0;

		m_pipeline = amalthea->m_device->CreateGraphicsPipeline(pipelineDesc);

		// Create Framebuffers
		for (AmaltheaFrame& ctx : amalthea->m_frames)
		{
			EuropaFramebufferCreateInfo desc;
			desc.attachments = { ctx.imageView, m_depthView };
			desc.width = amalthea->m_windowSize.x;
			desc.height = amalthea->m_windowSize.y;
			desc.layers = 1;
			desc.renderpass = m_mainRenderPass;

			EuropaFramebuffer::Ref framebuffer = amalthea->m_device->CreateFramebuffer(desc);

			m_frameBuffers.push_back(framebuffer);
		}

		// Constants & Descriptor Pools / Sets
		EuropaDescriptorPoolSizes descPoolSizes;
		descPoolSizes.UniformDynamic = 1 * amalthea->m_frames.size();
		descPoolSizes.UniformTexel = 3 * amalthea->m_frames.size();
		descPoolSizes.Storage = 3 * amalthea->m_frames.size();

		m_descPool = amalthea->m_device->CreateDescriptorPool(descPoolSizes, uint32(amalthea->m_frames.size()));

		for (uint32 i = 0; i < amalthea->m_frames.size(); i++)
		{
			m_descSets.push_back(m_descPool->AllocateDescriptorSet(descLayout));
		}

		m_constantsSize = alignUp(uint32(sizeof(ShaderConstants)), amalthea->m_device->GetMinUniformBufferOffsetAlignment());
	};

	AmaltheaBehaviors::OnDestroySwapChain f_onDestroySwapChain = [&](Amalthea* amalthea)
	{
		m_frameBuffers.clear();
		m_descSets.clear();
	};

	AmaltheaBehaviors::OnRender f_onRender = [&](Amalthea* amalthea, AmaltheaFrame& ctx, float time, float deltaTime)
	{
		if (amalthea->m_ioSurface->IsKeyDown('W'))
			m_orbitHeight += deltaTime * 0.5f;
		if (amalthea->m_ioSurface->IsKeyDown('S'))
			m_orbitHeight -= deltaTime * 0.5f;

		if (amalthea->m_ioSurface->IsKeyDown('E'))
			m_orbitRadius += deltaTime;
		if (amalthea->m_ioSurface->IsKeyDown('Q'))
			m_orbitRadius -= deltaTime;

		if (amalthea->m_ioSurface->IsKeyDown('A'))
			m_orbitAngle += deltaTime * 3.1415926f * 0.5;
		if (amalthea->m_ioSurface->IsKeyDown('D'))
			m_orbitAngle -= deltaTime * 3.1415926f * 0.5;

		auto constantsHandle = amalthea->m_streamingBuffer->AllocateTransient(m_constantsSize);
		ShaderConstants* constants = constantsHandle.Map<ShaderConstants>();

		constants->viewMtx = glm::lookAt(glm::vec3(cos(m_orbitAngle) * m_orbitRadius, m_orbitHeight, sin(m_orbitAngle) * m_orbitRadius), glm::vec3(0.0, 0.5, 0.0), glm::vec3(0.0, 1.0, 0.0));
		constants->projMtx = glm::perspective(glm::radians(60.0f), float(amalthea->m_windowSize.x) / (amalthea->m_windowSize.y), 0.01f, 256.0f);

		constants->projMtx[1].y = -constants->projMtx[1].y;

		constants->viewInvMtx = glm::inverse(constants->viewMtx);
		constants->projInvMtx = glm::inverse(constants->projMtx);

		constants->numLights = lights.size();
		constants->numTriangles = indices.size() / 3;

		constantsHandle.Unmap();

		m_descSets[ctx.frameIndex]->SetUniformBufferDynamic(constantsHandle.buffer, 0, constantsHandle.offset + m_constantsSize, 0, 0);
		m_descSets[ctx.frameIndex]->SetStorage(m_lightsBuffer, 0, uint32(lights.size() * sizeof(Light)), 1, 0);
		m_descSets[ctx.frameIndex]->SetStorage(m_vertexBuffer, 0, uint32(vertices.size() * sizeof(Vertex)), 2, 0);
		m_descSets[ctx.frameIndex]->SetStorage(m_indexBuffer, 0, uint32(indices.size() * sizeof(uint16)), 3, 0);

		EuropaClearValue clearValue[2];
		clearValue[0].color = glm::vec4(0.0, 0.0, 0.0, 1.0);
		clearValue[1].depthStencil = glm::vec2(1.0, 0.0);

		ctx.cmdlist->BeginRenderpass(m_mainRenderPass, m_frameBuffers[ctx.frameIndex], glm::ivec2(0), glm::uvec2(amalthea->m_windowSize), 2, clearValue);
		ctx.cmdlist->BindPipeline(m_pipeline);
		ctx.cmdlist->BindDescriptorSetsDynamicOffsets(EuropaPipelineBindPoint::Graphics, m_pipelineLayout, m_descSets[ctx.frameIndex], 0, constantsHandle.offset);
		ctx.cmdlist->DrawInstanced(6, 1, 0, 0);
		ctx.cmdlist->EndRenderpass();

		m_fps = m_fps * 0.7 + 0.3 * glm::clamp(1.0 / deltaTime, m_fps - 10.0, m_fps + 10.0);
		m_frameTimeLog.AddPoint(time, deltaTime * 1000.0);
		m_frameRateLog.AddPoint(time, m_fps);
		m_frameCount++;

		if (ImGui::Begin("Debug Info"))
		{
			ImGui::LabelText("", "CPU: %f ms", deltaTime * 1000.0);
			ImGui::LabelText("", "FPS: %f", m_fps);

			ImPlot::SetNextPlotLimitsX(time - 5.0, time, ImGuiCond_Always);
			ImPlot::SetNextPlotLimitsY(0.0, 40.0, ImGuiCond_Once, 0);
			ImPlot::SetNextPlotLimitsY(0.0, 160.0, ImGuiCond_Once, 1);
			// ImPlot::FitNextPlotAxes(false, true, true, false);
			// ImPlot::FitNextPlotAxes(false, true, true, false);
			if (ImPlot::BeginPlot("Performance", "Time", nullptr, ImVec2(-1, 0), ImPlotFlags_Default | ImPlotFlags_YAxis2)) {
				ImPlot::SetPlotYAxis(0);
				ImPlot::PlotLine("FrameTime", m_frameTimeLog.GetDataX(), m_frameTimeLog.GetDataY(), m_frameTimeLog.GetSize(), m_frameTimeLog.GetOffset());
				ImPlot::SetPlotYAxis(1);
				ImPlot::PlotLine("FPS", m_frameRateLog.GetDataX(), m_frameRateLog.GetDataY(), m_frameRateLog.GetSize(), m_frameRateLog.GetOffset());
				ImPlot::EndPlot();
			}
		}
		ImGui::End();
	};

	~TestApp()
	{
	}

	TestApp(GanymedeECS& ecs, Europa& e, IoSurface::Ref s)
		: m_amalthea(ecs, e, s)
	{
		ecs.RegisterHandler(m_amalthea.m_events.OnCreateDevice, &f_onCreateDevice);
		ecs.RegisterHandler(m_amalthea.m_events.OnDestroyDevice, &f_onDestroyDevice);
		ecs.RegisterHandler(m_amalthea.m_events.OnCreateSwapChain, &f_onCreateSwapChain);
		ecs.RegisterHandler(m_amalthea.m_events.OnDestroySwapChain, &f_onDestroySwapChain);
		ecs.RegisterHandler(m_amalthea.m_events.OnRender, &f_onRender);

		m_amalthea.Run();
	}
};

int AppMain(IoSurface::Ref s)
{
	// Test ECS
	GanymedeECS ecs;

	uint32 ev0 = ecs.RegisterEvent();
	uint32 ev1 = ecs.RegisterEvent();

	std::function<void()> f0 = []() { GanymedePrint "hahaha"; };
	std::function<void(int)> f1 = [](int v) { GanymedePrint "A signaled with", v; };
	std::function<void(int)> f2 = [&](int v) { GanymedePrint "B signaled with", v, & ecs; };
	std::function<void(int)> f3 = [](int v) { GanymedePrint "C signaled with", v; };

	ecs.RegisterHandler(ev0, &f0);
	ecs.RegisterHandler(ev1, &f1);
	ecs.RegisterHandler(ev1, &f2);
	ecs.RegisterHandler(ev1, &f3);

	ecs.Signal(ev0);
	ecs.Signal(ev1, 40);

	// Create Europa Instance
	Europa& europa = EuropaVk();

	TestApp app(ecs, europa, s);

	return 0;
}