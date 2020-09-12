#include "Io/Source/Io.h"
#include "Io/Source/IoEntryFunc.h"
#include "Europa/Source/EuropaVk.h"
#include "Amalthea/Source/Amalthea.h"
#include "Ganymede/Source/Ganymede.h"
#include "Ganymede/Source/GanymedeECS.h"
#include "Himalia/Source/Himalia.h"

#include "trace.comp.h"
#include "trace_speculative.comp.h"
#include "launch.comp.h"
#include "raysort.comp.h"
#include "visualize.frag.h"
#include "visualize.vert.h"
#include "composite.frag.h"
#include "composite.vert.h"

#include <thread>
#include <chrono>

#include <glm/gtx/transform.hpp>
#include <glm/gtc/type_precision.hpp>

#include "blueNoise.h"

#include "ShaderData.h"
#include "BVH.h"

std::vector<glm::vec4> vertexPosition;
std::vector<VertexAux> vertexAuxilary;
std::vector<uint32> indices;
std::vector<Light> lights = {
	{ glm::vec3(0.0, 1.4, 0.0), glm::vec3(1.0, 1.0, 1.0) },
};
std::vector<BVHNode> nodes;

uint32 bvhVisStartVertex = 0;
uint32 bvhVisStartIndex = 0;

class TestApp
{
public:
	Amalthea m_amalthea;

	EuropaBuffer::Ref m_vertexPosBuffer;
	EuropaBufferView::Ref m_vertexPosBufferView;
	EuropaBuffer::Ref m_vertexBuffer;
	EuropaBuffer::Ref m_indexBuffer;
	EuropaBufferView::Ref m_indexBufferView;
	EuropaBuffer::Ref m_lightsBuffer;
	EuropaBuffer::Ref m_blueNoiseBuffer;
	EuropaBuffer::Ref m_bvhBuffer;
	EuropaBuffer::Ref m_rayStackBuffer;
	EuropaBuffer::Ref m_jobBuffer;

	EuropaImage::Ref m_depthImage;
	EuropaImageView::Ref m_depthView;

	EuropaRenderPass::Ref m_mainRenderPass;

	EuropaDescriptorPool::Ref m_descPool;
	EuropaPipeline::Ref m_pipeline;
	EuropaPipeline::Ref m_pipelineSpeculative;
	EuropaPipeline::Ref m_pipelineRayLaunch;
	EuropaPipeline::Ref m_pipelineRaySort;
	EuropaPipeline::Ref m_pipelineComposite;
	EuropaPipeline::Ref m_pipelineVis;
	EuropaPipeline::Ref m_pipelineVisLine;
	EuropaPipelineLayout::Ref m_pipelineLayout;

	std::vector<EuropaDescriptorSet::Ref> m_descSets;
	std::vector<EuropaFramebuffer::Ref> m_frameBuffers;

	std::vector<EuropaImage::Ref> m_accumulationImages;
	std::vector<EuropaImageView::Ref> m_accumulationImageViews;

	uint32 m_frameIndex = 0;
	uint32 m_maxDepth = 5;
	uint32 m_constantsSize;
	bool m_visualize = false;
	bool m_raySort = true;

	glm::vec3 m_focusCenter = glm::vec3(0.0, 0.0, 0.0);
	float m_orbitHeight = 0.5;
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

		//plyModel.LoadFile("../Models/CBbunny.ply");
		//plyModel.LoadFile("../Models/CBdragon.ply");
		//plyModel.LoadFile("../Models/CBmonkey.ply");
		//plyModel.LoadFile("../Models/minecraft.ply");
		//plyModel.LoadFile("../Models/cornellBox.ply");
		//plyModel.LoadFile("../Models/sponza.ply");
		//plyModel.LoadFile("../Models/conference.ply");
		//plyModel.LoadFile("../Models/livingRoom.ply");
		plyModel.LoadFile("../Models/SanMiguel.ply");

		HimaliaVertexProperty vertexFormatAux[] = {
			HimaliaVertexProperty::Normal,
			HimaliaVertexProperty::ColorRGBA8
		};
		uint32 alignments[] = {
			0, offsetof(VertexAux, VertexAux::color)
		};
		plyModel.mesh.BuildVertices<VertexAux>(vertexAuxilary, 2, vertexFormatAux, alignments);
		
		HimaliaVertexProperty vertexFormat = HimaliaVertexProperty::Position;
		plyModel.mesh.BuildVertices<glm::vec4>(vertexPosition, 1, &vertexFormat);
		
		plyModel.mesh.BuildIndices<uint32>(indices);

		bvhVisStartIndex = uint32(indices.size());
		bvhVisStartVertex = uint32(vertexPosition.size());

		nodes = BuildBVH(vertexPosition, indices);

		VisualizeBVH(nodes, vertexPosition, vertexAuxilary, indices);

		// Create & Upload geometry buffers
		EuropaBufferInfo vertexBufferInfo;
		vertexBufferInfo.exclusive = true;
		vertexBufferInfo.size = uint32(vertexAuxilary.size() * sizeof(VertexAux));
		vertexBufferInfo.usage = EuropaBufferUsage(EuropaBufferUsageStorage | EuropaBufferUsageVertex | EuropaBufferUsageTransferDst);
		vertexBufferInfo.memoryUsage = EuropaMemoryUsage::GpuOnly;
		m_vertexBuffer = amalthea->m_device->CreateBuffer(vertexBufferInfo);

		amalthea->m_transferUtil->UploadToBufferEx(m_vertexBuffer, vertexAuxilary.data(), uint32(vertexAuxilary.size()));

		EuropaBufferInfo vertexBufferPosInfo;
		vertexBufferPosInfo.exclusive = true;
		vertexBufferPosInfo.size = uint32(vertexPosition.size() * sizeof(glm::vec4));
		vertexBufferPosInfo.usage = EuropaBufferUsage(EuropaBufferUsageUniformTexel | EuropaBufferUsageVertex | EuropaBufferUsageTransferDst);
		vertexBufferPosInfo.memoryUsage = EuropaMemoryUsage::GpuOnly;
		m_vertexPosBuffer = amalthea->m_device->CreateBuffer(vertexBufferPosInfo);

		amalthea->m_transferUtil->UploadToBufferEx(m_vertexPosBuffer, vertexPosition.data(), uint32(vertexPosition.size()));

		m_vertexPosBufferView = amalthea->m_device->CreateBufferView(m_vertexPosBuffer, vertexBufferPosInfo.size, 0, EuropaImageFormat::RGBA32F);

		EuropaBufferInfo indexBufferInfo;
		indexBufferInfo.exclusive = true;
		indexBufferInfo.size = uint32(indices.size() * sizeof(uint32));
		indexBufferInfo.usage = EuropaBufferUsage(EuropaBufferUsageUniformTexel | EuropaBufferUsageIndex | EuropaBufferUsageTransferDst);
		indexBufferInfo.memoryUsage = EuropaMemoryUsage::GpuOnly;
		m_indexBuffer = amalthea->m_device->CreateBuffer(indexBufferInfo);

		m_indexBufferView = amalthea->m_device->CreateBufferView(m_indexBuffer, uint32(bvhVisStartIndex * sizeof(uint32)), 0, EuropaImageFormat::RGB32UI);

		amalthea->m_transferUtil->UploadToBufferEx(m_indexBuffer, indices.data(), uint32(indices.size()));

		EuropaBufferInfo lightBufferInfo;
		lightBufferInfo.exclusive = true;
		lightBufferInfo.size = uint32(lights.size() * sizeof(Light));
		lightBufferInfo.usage = EuropaBufferUsage(EuropaBufferUsageStorage | EuropaBufferUsageTransferDst);
		lightBufferInfo.memoryUsage = EuropaMemoryUsage::GpuOnly;
		m_lightsBuffer = amalthea->m_device->CreateBuffer(lightBufferInfo);

		amalthea->m_transferUtil->UploadToBufferEx(m_lightsBuffer, lights.data(), uint32(lights.size()));

		EuropaBufferInfo blueNoiseBufferInfo;
		blueNoiseBufferInfo.exclusive = true;
		blueNoiseBufferInfo.size = sizeof(_blueNoise);
		blueNoiseBufferInfo.usage = EuropaBufferUsage(EuropaBufferUsageStorage | EuropaBufferUsageTransferDst);
		blueNoiseBufferInfo.memoryUsage = EuropaMemoryUsage::GpuOnly;
		m_blueNoiseBuffer = amalthea->m_device->CreateBuffer(blueNoiseBufferInfo);

		amalthea->m_transferUtil->UploadToBufferEx(m_blueNoiseBuffer, _blueNoise, sizeof(_blueNoise) / sizeof(uint16));

		EuropaBufferInfo bvhBufferInfo;
		bvhBufferInfo.exclusive = true;
		bvhBufferInfo.size = uint32(nodes.size() * sizeof(BVHNode));
		bvhBufferInfo.usage = EuropaBufferUsage(EuropaBufferUsageStorage | EuropaBufferUsageTransferDst);
		bvhBufferInfo.memoryUsage = EuropaMemoryUsage::GpuOnly;
		m_bvhBuffer = amalthea->m_device->CreateBuffer(bvhBufferInfo);

		amalthea->m_transferUtil->UploadToBufferEx(m_bvhBuffer, nodes.data(), uint32(nodes.size()));

		amalthea->m_ioSurface->SetKeyCallback([](uint8 keyAscii, uint16 keyV, std::string, IoKeyboardEvent ev)
			{
				GanymedePrint "Key", keyAscii, keyV, IoKeyboardEventToString(ev);
			});
	};

	AmaltheaBehaviors::OnDestroyDevice f_onDestroyDevice = [&](Amalthea* amalthea)
	{
	};

	AmaltheaBehaviors::OnCreateSwapChain f_onCreateSwapChain = [&](Amalthea* amalthea)
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

		// Create Accumulation buffer
		m_accumulationImages.clear();
		m_accumulationImageViews.clear();

		for (int i = 0; i < amalthea->m_frames.size(); i++)
		{
			EuropaImageInfo info;
			info.width = amalthea->m_windowSize.x;
			info.height = amalthea->m_windowSize.y;
			info.initialLayout = EuropaImageLayout::General;
			info.type = EuropaImageType::Image2D;
			info.format = EuropaImageFormat::RGBA32F;
			info.usage = EuropaImageUsage(EuropaImageUsageStorage | EuropaImageUsageTransferSrc | EuropaImageUsageTransferDst);
			info.memoryUsage = EuropaMemoryUsage::GpuOnly;

			auto image = amalthea->m_device->CreateImage(info);
			m_accumulationImages.push_back(image);

			EuropaImageViewCreateInfo viewInfo;
			viewInfo.format = EuropaImageFormat::RGBA32F;
			viewInfo.image = image;
			viewInfo.type = EuropaImageViewType::View2D;
			viewInfo.minArrayLayer = 0;
			viewInfo.minMipLevel = 0;
			viewInfo.numArrayLayers = 1;
			viewInfo.numMipLevels = 1;
		
			m_accumulationImageViews.push_back(amalthea->m_device->CreateImageView(viewInfo));
		}

		// Create Ray Stack
		EuropaBufferInfo rayStackInfo;
		rayStackInfo.exclusive = true;
		rayStackInfo.size = uint32(amalthea->m_windowSize.x * amalthea->m_windowSize.y * m_maxDepth * sizeof(RayStack));
		rayStackInfo.usage = EuropaBufferUsage(EuropaBufferUsageStorage);
		rayStackInfo.memoryUsage = EuropaMemoryUsage::GpuOnly;
		m_rayStackBuffer = amalthea->m_device->CreateBuffer(rayStackInfo);

		EuropaBufferInfo jobBufferInfo;
		jobBufferInfo.exclusive = true;
		jobBufferInfo.size = uint32(amalthea->m_windowSize.x * amalthea->m_windowSize.y * sizeof(RayJob));
		jobBufferInfo.usage = EuropaBufferUsage(EuropaBufferUsageStorage);
		jobBufferInfo.memoryUsage = EuropaMemoryUsage::GpuOnly;
		m_jobBuffer = amalthea->m_device->CreateBuffer(jobBufferInfo);

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
		EuropaAttachmentReference depthAttachment = { depthTarget, EuropaImageLayout::DepthStencilAttachment };
		std::vector<EuropaAttachmentReference> attachmentsForward = {
			{ presentTarget, EuropaImageLayout::ColorAttachment }
		};
		uint32 forwardPass = m_mainRenderPass->AddSubpass(EuropaPipelineBindPoint::Graphics, attachmentsForward, &depthAttachment);
		m_mainRenderPass->AddDependency(EuropaRenderPass::SubpassExternal, forwardPass, EuropaPipelineStageBottomOfPipe, EuropaAccessNone, EuropaPipelineStageFragmentShader, EuropaAccessColorAttachmentWrite);
		m_mainRenderPass->CreateRenderpass();

		// Create Pipeline
		EuropaDescriptorSetLayout::Ref descLayout = amalthea->m_device->CreateDescriptorSetLayout();
		descLayout->DynamicUniformBuffer(0, 1, EuropaShaderStageAll);
		descLayout->Storage(1, 1, EuropaShaderStageCompute);
		descLayout->Storage(2, 1, EuropaShaderStageCompute);
		descLayout->BufferViewUniform(3, 1, EuropaShaderStageCompute);
		descLayout->Storage(4, 1, EuropaShaderStageCompute);
		descLayout->ImageViewStorage(5, 1, EuropaShaderStageAll);
		descLayout->ImageViewStorage(6, 1, EuropaShaderStageAll);
		descLayout->BufferViewUniform(7, 1, EuropaShaderStageCompute);
		descLayout->Storage(8, 1, EuropaShaderStageCompute);
		descLayout->Storage(9, 1, EuropaShaderStageAll);
		descLayout->Storage(10, 1, EuropaShaderStageCompute);
		descLayout->Build();

		m_pipelineLayout = amalthea->m_device->CreatePipelineLayout(EuropaPipelineLayoutInfo{ 1, 0, &descLayout });

		{
			EuropaShaderModule::Ref shader = amalthea->m_device->CreateShaderModule(shader_spv_trace_comp_h, sizeof(shader_spv_trace_comp_h));

			EuropaShaderStageInfo stage = { EuropaShaderStageCompute, shader, "main" };

			m_pipeline = amalthea->m_device->CreateComputePipeline(stage, m_pipelineLayout);
		}

		{
			EuropaShaderModule::Ref shader = amalthea->m_device->CreateShaderModule(shader_spv_trace_speculative_comp_h, sizeof(shader_spv_trace_speculative_comp_h));

			EuropaShaderStageInfo stage = { EuropaShaderStageCompute, shader, "main" };

			m_pipelineSpeculative = amalthea->m_device->CreateComputePipeline(stage, m_pipelineLayout);
		}

		{
			EuropaShaderModule::Ref shader = amalthea->m_device->CreateShaderModule(shader_spv_launch_comp_h, sizeof(shader_spv_launch_comp_h));

			EuropaShaderStageInfo stage = { EuropaShaderStageCompute, shader, "main" };

			m_pipelineRayLaunch = amalthea->m_device->CreateComputePipeline(stage, m_pipelineLayout);
		}

		{
			EuropaShaderModule::Ref shader = amalthea->m_device->CreateShaderModule(shader_spv_raysort_comp_h, sizeof(shader_spv_raysort_comp_h));

			EuropaShaderStageInfo stage = { EuropaShaderStageCompute, shader, "main" };

			m_pipelineRaySort = amalthea->m_device->CreateComputePipeline(stage, m_pipelineLayout);
		}

		{
			EuropaShaderModule::Ref shaderFragment = amalthea->m_device->CreateShaderModule(shader_spv_composite_frag_h, sizeof(shader_spv_composite_frag_h));
			EuropaShaderModule::Ref shaderVertex = amalthea->m_device->CreateShaderModule(shader_spv_composite_vert_h, sizeof(shader_spv_composite_vert_h));

			EuropaShaderStageInfo stages[2] = {
				EuropaShaderStageInfo{ EuropaShaderStageFragment, shaderFragment, "main" },
				EuropaShaderStageInfo{ EuropaShaderStageVertex, shaderVertex, "main"}
			};

			EuropaGraphicsPipelineCreateInfo pipelineDesc{};

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
			pipelineDesc.depthStencil.enableDepthTest = false;
			pipelineDesc.depthStencil.enableDepthWrite = false;
			pipelineDesc.layout = m_pipelineLayout;
			pipelineDesc.renderpass = m_mainRenderPass;
			pipelineDesc.targetSubpass = forwardPass;

			m_pipelineComposite = amalthea->m_device->CreateGraphicsPipeline(pipelineDesc);
		}

		{
			EuropaShaderModule::Ref shaderFragment = amalthea->m_device->CreateShaderModule(shader_spv_visualize_frag_h, sizeof(shader_spv_visualize_frag_h));
			EuropaShaderModule::Ref shaderVertex = amalthea->m_device->CreateShaderModule(shader_spv_visualize_vert_h, sizeof(shader_spv_visualize_vert_h));

			EuropaShaderStageInfo stages[2] = {
				EuropaShaderStageInfo{ EuropaShaderStageFragment, shaderFragment, "main" },
				EuropaShaderStageInfo{ EuropaShaderStageVertex, shaderVertex, "main"}
			};

			EuropaGraphicsPipelineCreateInfo pipelineDesc{};

			EuropaVertexInputBindingInfo binding[2];
			binding[0].binding = 0;
			binding[0].stride = sizeof(glm::vec4);
			binding[0].perInstance = false;

			binding[1].binding = 1;
			binding[1].stride = sizeof(VertexAux);
			binding[1].perInstance = false;

			EuropaVertexAttributeBindingInfo attributes[3];
			attributes[0].binding = 0;
			attributes[0].location = 0;
			attributes[0].offset = 0;
			attributes[0].format = EuropaImageFormat::RGB32F;

			attributes[1].binding = 1;
			attributes[1].location = 1;
			attributes[1].offset = offsetof(VertexAux, VertexAux::color);
			attributes[1].format = EuropaImageFormat::RGBA8Unorm;

			attributes[2].binding = 1;
			attributes[2].location = 2;
			attributes[2].offset = offsetof(VertexAux, VertexAux::normal);
			attributes[2].format = EuropaImageFormat::RGB32F;

			pipelineDesc.shaderStageCount = 2;
			pipelineDesc.stages = stages;
			pipelineDesc.vertexInput.vertexBindingCount = 2;
			pipelineDesc.vertexInput.vertexBindings = binding;
			pipelineDesc.vertexInput.attributeBindingCount = 3;
			pipelineDesc.vertexInput.attributeBindings = attributes;
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

			pipelineDesc.targetSubpass = forwardPass;
			pipelineDesc.inputAssembly.topology = EuropaPrimitiveTopology::TriangleList;
			m_pipelineVis = amalthea->m_device->CreateGraphicsPipeline(pipelineDesc);

			pipelineDesc.targetSubpass = forwardPass;
			pipelineDesc.inputAssembly.topology = EuropaPrimitiveTopology::LineList;
			m_pipelineVisLine = amalthea->m_device->CreateGraphicsPipeline(pipelineDesc);
		}

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
		descPoolSizes.UniformDynamic = uint32(1 * amalthea->m_frames.size());
		descPoolSizes.UniformTexel = uint32(2 * amalthea->m_frames.size());
		descPoolSizes.StorageImage = uint32(2 * amalthea->m_frames.size());
		descPoolSizes.Storage = uint32(7 * amalthea->m_frames.size());

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
		bool clear = false;

		if (amalthea->m_ioSurface->IsKeyDown('W'))
		{
			m_orbitHeight += deltaTime * 0.5f;
			clear = true;
		}

		if (amalthea->m_ioSurface->IsKeyDown('S'))
		{
			m_orbitHeight -= deltaTime * 0.5f;
			clear = true;
		}

		if (amalthea->m_ioSurface->IsKeyDown('E'))
		{
			m_orbitRadius += deltaTime;
			clear = true;
		}
		
		if (amalthea->m_ioSurface->IsKeyDown('Q'))
		{
			m_orbitRadius -= deltaTime;
			clear = true;
		}

		if (amalthea->m_ioSurface->IsKeyDown('A'))
		{
			m_orbitAngle += deltaTime * 3.1415926f * 0.5f;
			clear = true;
		}
	
		if (amalthea->m_ioSurface->IsKeyDown('D'))
		{
			m_orbitAngle -= deltaTime * 3.1415926f * 0.5f;
			clear = true;
		}

		m_frameIndex = (m_frameIndex + 1) & 0xFFFF;
		
		auto constantsHandle = amalthea->m_streamingBuffer->AllocateTransient(m_constantsSize);
		ShaderConstants* constants = constantsHandle.Map<ShaderConstants>();

		constants->viewMtx = glm::lookAt(glm::vec3(cos(m_orbitAngle) * m_orbitRadius, m_orbitHeight, sin(m_orbitAngle) * m_orbitRadius), m_focusCenter, glm::vec3(0.0, 1.0, 0.0));
		constants->projMtx = glm::perspective(glm::radians(60.0f), float(amalthea->m_windowSize.x) / (amalthea->m_windowSize.y), 0.01f, 256.0f);

		constants->projMtx[1].y = -constants->projMtx[1].y;

		constants->viewInvMtx = glm::inverse(constants->viewMtx);
		constants->projInvMtx = glm::inverse(constants->projMtx);

		constants->viewportSize = glm::vec2(amalthea->m_windowSize);

		constants->numLights = uint32(lights.size());
		constants->numTriangles = bvhVisStartIndex / 3;
		constants->frameIndex = m_frameIndex;
		constants->numRays = m_maxDepth;
		constants->numBVHNodes = uint32(nodes.size());

		constantsHandle.Unmap();

		m_descSets[ctx.frameIndex]->SetUniformBufferDynamic(constantsHandle.buffer, 0, constantsHandle.offset + m_constantsSize, 0, 0);
		if (!m_visualize)
		{
			m_descSets[ctx.frameIndex]->SetStorage(m_lightsBuffer, 0, uint32(lights.size() * sizeof(Light)), 1, 0);
			m_descSets[ctx.frameIndex]->SetStorage(m_vertexBuffer, 0, uint32(bvhVisStartVertex * sizeof(VertexAux)), 2, 0);
			m_descSets[ctx.frameIndex]->SetBufferViewUniform(m_indexBufferView, 3, 0);
			m_descSets[ctx.frameIndex]->SetStorage(m_blueNoiseBuffer, 0, sizeof(_blueNoise), 4, 0);
			m_descSets[ctx.frameIndex]->SetImageViewStorage(m_accumulationImageViews[(ctx.frameIndex - 1) % amalthea->m_frames.size()], EuropaImageLayout::General, 5, 0);
			m_descSets[ctx.frameIndex]->SetImageViewStorage(m_accumulationImageViews[ctx.frameIndex], EuropaImageLayout::General, 6, 0);
			m_descSets[ctx.frameIndex]->SetBufferViewUniform(m_vertexPosBufferView, 7, 0);
			m_descSets[ctx.frameIndex]->SetStorage(m_bvhBuffer, 0, uint32(nodes.size() * sizeof(BVHNode)), 8, 0);
			m_descSets[ctx.frameIndex]->SetStorage(m_rayStackBuffer, 0, uint32(amalthea->m_windowSize.x * amalthea->m_windowSize.y * m_maxDepth * sizeof(RayStack)), 9, 0);
			m_descSets[ctx.frameIndex]->SetStorage(m_jobBuffer, 0, uint32(amalthea->m_windowSize.x * amalthea->m_windowSize.y * sizeof(RayJob)), 10, 0);
		}

		EuropaClearValue clearValue[2];
		clearValue[0].color = glm::vec4(0.0, 0.0, 0.0, 1.0);
		clearValue[1].depthStencil = glm::vec2(1.0, 0.0);

		if (!m_visualize)
		{
			ctx.cmdlist->BindDescriptorSetsDynamicOffsets(EuropaPipelineBindPoint::Compute, m_pipelineLayout, m_descSets[ctx.frameIndex], 0, constantsHandle.offset);
			ctx.cmdlist->BindCompute(m_pipelineRayLaunch);
			ctx.cmdlist->Dispatch(uint32(ceil(float(amalthea->m_windowSize.x) / 32.0f)), uint32(ceil(float(amalthea->m_windowSize.y) / 32.0f)), 1);
			
			for (uint32 d = 0; d < m_maxDepth; d++)
			{
				ctx.cmdlist->Barrier(
					m_rayStackBuffer, uint32(amalthea->m_windowSize.x * amalthea->m_windowSize.y * m_maxDepth * sizeof(RayStack)), 0,
					EuropaAccessShaderWrite, EuropaAccessShaderRead,
					EuropaPipelineStageComputeShader, EuropaPipelineStageComputeShader
				);
				
				if (d == 0)
					ctx.cmdlist->BindCompute(m_pipelineSpeculative);
				else
					ctx.cmdlist->BindCompute(m_pipeline);

				ctx.cmdlist->Dispatch(uint32(ceil(float(amalthea->m_windowSize.x) / 8.0f)), uint32(ceil(float(amalthea->m_windowSize.y) / 8.0f)), 1);

				if (m_raySort && d != m_maxDepth - 1)
				{
					ctx.cmdlist->Barrier(
						m_jobBuffer, uint32(amalthea->m_windowSize.x * amalthea->m_windowSize.y * sizeof(RayJob)), 0,
						EuropaAccessShaderWrite, EuropaAccessShaderRead,
						EuropaPipelineStageComputeShader, EuropaPipelineStageComputeShader
					); 
					ctx.cmdlist->BindCompute(m_pipelineRaySort);
					ctx.cmdlist->Dispatch(uint32(ceil(float(amalthea->m_windowSize.x) / 256.0f)), uint32(amalthea->m_windowSize.y), 1);
					ctx.cmdlist->Barrier(
						m_jobBuffer, uint32(amalthea->m_windowSize.x * amalthea->m_windowSize.y * sizeof(RayJob)), 0,
						EuropaAccessShaderWrite, EuropaAccessShaderRead,
						EuropaPipelineStageComputeShader, EuropaPipelineStageComputeShader
					); 
				}
			}
			
			ctx.cmdlist->Barrier(
				m_accumulationImages[ctx.frameIndex],
				EuropaAccessNone, EuropaAccessShaderRead, EuropaImageLayout::General, EuropaImageLayout::General,
				EuropaPipelineStageBottomOfPipe, EuropaPipelineStageFragmentShader
			);
		}

		ctx.cmdlist->BeginRenderpass(m_mainRenderPass, m_frameBuffers[ctx.frameIndex], glm::ivec2(0), glm::uvec2(amalthea->m_windowSize), 2, clearValue);
		ctx.cmdlist->BindDescriptorSetsDynamicOffsets(EuropaPipelineBindPoint::Graphics, m_pipelineLayout, m_descSets[ctx.frameIndex], 0, constantsHandle.offset);
		if (m_visualize)
		{
			ctx.cmdlist->BindPipeline(m_pipelineVis);
			ctx.cmdlist->BindVertexBuffer(m_vertexPosBuffer, 0, 0);
			ctx.cmdlist->BindVertexBuffer(m_vertexBuffer, 0, 1);
			ctx.cmdlist->BindIndexBuffer(m_indexBuffer, 0, EuropaImageFormat::R32UI);
			ctx.cmdlist->DrawIndexed(bvhVisStartIndex, 1, 0, 0, 0);
			ctx.cmdlist->BindPipeline(m_pipelineVisLine);
			ctx.cmdlist->DrawIndexed(uint32(indices.size()) - bvhVisStartIndex, 1, bvhVisStartIndex, bvhVisStartVertex, 0);
		}
		else
		{
			ctx.cmdlist->BindPipeline(m_pipelineComposite);
			ctx.cmdlist->DrawInstanced(6, 1, 0, 0);
		}
		ctx.cmdlist->EndRenderpass();

		m_fps = m_fps * 0.7f + 0.3f * glm::clamp(1.0f / deltaTime, m_fps - 10.0f, m_fps + 10.0f);
		m_frameTimeLog.AddPoint(time, deltaTime * 1000.0f);
		m_frameRateLog.AddPoint(time, m_fps);
		m_frameCount++;

		if (ImGui::Begin("Debug Info"))
		{
			ImGui::LabelText("", "CPU: %f ms", deltaTime * 1000.0);
			ImGui::LabelText("", "FPS: %f", m_fps);

			ImGui::DragFloat3("Center", &m_focusCenter.x);
			
			ImGui::SliderInt("Max Depth", (int*)&m_maxDepth, 1, 5);

			ImGui::Checkbox("Ray Sorting", &m_raySort);
			ImGui::SameLine();
			ImGui::Checkbox("Visualization", &m_visualize);
			ImGui::SameLine();
			if (ImGui::Button("Reset Image")) clear = true;

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

		if (clear)
		{
			for (auto i : m_accumulationImages)
			{
				ctx.cmdlist->ClearImage(i, EuropaImageLayout::General, glm::vec4(0.0));
			}
		}
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
	GanymedeECS ecs;

	// Create Europa Instance
	Europa& europa = EuropaVk();

	TestApp app(ecs, europa, s);

	return 0;
}