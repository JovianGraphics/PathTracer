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
#include <atomic>
#include <chrono>
#include <fstream>

extern "C"
{
#include "miniz.h"
}

#include <glm/gtx/transform.hpp>
#include <glm/gtc/type_precision.hpp>

#include "blueNoise.h"

#include "ShaderData.h"
#include "BVH.h"

#include "ImGuiExtensions.h"

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

	// GFX data
	struct {
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

		EuropaImage::Ref m_accumulationImages;
		EuropaImageView::Ref m_accumulationImageViews;
		EuropaImage::Ref m_currentImage;
		EuropaImageView::Ref m_currentImageView;

		std::vector<EuropaBuffer::Ref> m_currentImageCpuBuffers;
	};

	// Options & settings
	struct {
		uint32 m_frameIndex = 0;
		uint32 m_maxDepth = 5;
		uint32 m_constantsSize;
		bool m_visualize = false;
		bool m_raySort = true;
		bool m_dumpData = false;
	};

	// Scene parameters
	struct {
		bool sceneLoaded = false;

		std::string m_sceneFile = "../Models/cubeBinary.ply";

		glm::vec3 m_focusCenter = glm::vec3(0.0, 0.0, 0.0);
		float m_orbitHeight = 0.5;
		float m_orbitRadius = 3.0;
		float m_orbitAngle = 0.0;

		glm::vec3 m_ambientRadiance = glm::vec3(0.4, 0.5, 0.7);
	};

	// Peformance trackers
	struct {
		GanymedeScrollingBuffer m_frameTimeLog = GanymedeScrollingBuffer(1000, 0);
		GanymedeScrollingBuffer m_frameRateLog = GanymedeScrollingBuffer(1000, 0);
		uint32 m_frameCount = 0;
		float m_fps = 0.0;
		float m_bvhBuildProgress = 0.0f;
	};

	void UpdateLights()
	{
		EuropaBufferInfo lightBufferInfo;
		lightBufferInfo.exclusive = true;
		lightBufferInfo.size = uint32(lights.size() * sizeof(Light));
		lightBufferInfo.usage = EuropaBufferUsage(EuropaBufferUsageStorage | EuropaBufferUsageTransferDst);
		lightBufferInfo.memoryUsage = EuropaMemoryUsage::GpuOnly;
		m_lightsBuffer = m_amalthea.m_device->CreateBuffer(lightBufferInfo);

		m_amalthea.m_transferUtil->UploadToBufferEx(m_lightsBuffer, lights.data(), uint32(lights.size()));
	}

	// Write-out related data / structures
	struct {
		std::vector<char*> writeBuffers;
		std::vector<uint32> writeIndex;
		std::vector<uint32> writeSizes;
		std::atomic<uint32> workInWriteQueue = 0;

		std::condition_variable writeJobAvailable;
		std::mutex writeJobQueueLock;

		std::vector<std::thread> workers;
		bool terminate = false;
	};

	void WriteWorker()
	{
		while (!terminate)
		{
			std::unique_lock<std::mutex> lk(writeJobQueueLock);
			writeJobAvailable.wait(lk, [&] { return terminate || writeBuffers.size() > 0; });

			if (writeBuffers.empty())
			{
				lk.unlock();
				if (terminate)
					break;
				else
					continue;
			}

			uint32 index = writeIndex.back();
			char* buffer = writeBuffers.back();
			uint32 size = writeSizes.back();

			GanymedePrint "Writing", index;

			writeIndex.pop_back();
			writeBuffers.pop_back();
			writeSizes.pop_back();

			lk.unlock();

			std::stringstream ss;
			ss << "RayStackBuffer_" << index << ".bin.z";

			mz_ulong compressedSizeEst = mz_compressBound(size);
			uint8* compressed = (uint8*) malloc(compressedSizeEst);
			mz_ulong compressedSize = compressedSizeEst;
			int result = mz_compress2(compressed, &compressedSize, (uint8*)buffer, size, MZ_DEFAULT_LEVEL);
			if (result != MZ_OK)
			{
				GanymedePrint "miniz returned", mz_error(result), '(', result, ')';
				throw std::runtime_error("miniz deflate error");
			}

			std::ofstream file;
			file.open(ss.str(), std::ofstream::out | std::ofstream::binary);
			file.write((char*)compressed, compressedSize);
			file.close();

			ss << ".txt";

			file.open(ss.str(), std::ofstream::out);
			file << m_amalthea.m_windowSize.x << "," << m_amalthea.m_windowSize.y << std::endl;
			file.close();

			free(compressed);
			free(buffer);

			workInWriteQueue -= 1;
		}
	};
	void WriteBuffer(EuropaBuffer::Ref buffer, uint32 index, uint32 size)
	{
		auto start = std::chrono::high_resolution_clock::now();

		char* mapped = buffer->Map<char>();
		char* copy = (char*)malloc(size);

		memcpy(copy, mapped, size);

		buffer->Unmap();

		auto end = std::chrono::high_resolution_clock::now();

		GanymedePrint "Copy finished", index, std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count() * 1000.0, "ms";

		{
			while (workInWriteQueue > 64)
			{
				std::this_thread::sleep_for(std::chrono::milliseconds(5));
			}
			std::lock_guard<std::mutex> lk(writeJobQueueLock);
			workInWriteQueue++;
			writeBuffers.push_back(copy);
			writeIndex.push_back(index);
			writeSizes.push_back(size);
			writeJobAvailable.notify_one();
		}
	}

	AmaltheaBehaviors::OnCreateDevice f_onCreateDevice = [&](Amalthea* amalthea)
	{
		m_bvhBuildProgress = 0.0f;

		// ASYNC loading
		std::thread loading_thread([&](Amalthea* amalthea) {
			// Load Model
			HimaliaPlyModel plyModel;

			plyModel.LoadFile(m_sceneFile);

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

			nodes = BuildBVH(vertexPosition, indices, m_bvhBuildProgress);

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

			UpdateLights();

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
		
			sceneLoaded = true;
		}, amalthea);

		loading_thread.detach();

		amalthea->m_ioSurface->SetKeyCallback([](uint8 keyAscii, uint16 keyV, std::string, IoKeyboardEvent ev)
			{
				GanymedePrint "Key", keyAscii, keyV, IoKeyboardEventToString(ev);
			});
	};

	AmaltheaBehaviors::OnDestroyDevice f_onDestroyDevice = [&](Amalthea* amalthea)
	{
		vertexPosition.clear(); vertexPosition.shrink_to_fit();
		vertexAuxilary.clear(); vertexAuxilary.shrink_to_fit();
		indices.clear(); indices.shrink_to_fit();

		sceneLoaded = false;
	};

	void ReloadScene()
	{
		f_onDestroyDevice(&m_amalthea);
		m_amalthea.m_cmdQueue->WaitIdle();
		m_amalthea.m_device->WaitIdle();
		f_onCreateDevice(&m_amalthea);
		m_amalthea.m_cmdQueue->WaitIdle();
		m_amalthea.m_device->WaitIdle();
	}

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
		m_accumulationImages = nullptr;
		m_accumulationImageViews = nullptr;

		{
			EuropaImageInfo info;
			info.width = m_amalthea.m_windowSize.x;
			info.height = m_amalthea.m_windowSize.y;
			info.initialLayout = EuropaImageLayout::General;
			info.type = EuropaImageType::Image2D;
			info.format = EuropaImageFormat::RGBA32F;
			info.usage = EuropaImageUsage(EuropaImageUsageStorage | EuropaImageUsageTransferSrc | EuropaImageUsageTransferDst);
			info.memoryUsage = EuropaMemoryUsage::GpuOnly;

			m_accumulationImages = m_amalthea.m_device->CreateImage(info);

			EuropaImageViewCreateInfo viewInfo;
			viewInfo.format = EuropaImageFormat::RGBA32F;
			viewInfo.image = m_accumulationImages;
			viewInfo.type = EuropaImageViewType::View2D;
			viewInfo.minArrayLayer = 0;
			viewInfo.minMipLevel = 0;
			viewInfo.numArrayLayers = 1;
			viewInfo.numMipLevels = 1;

			m_accumulationImageViews = m_amalthea.m_device->CreateImageView(viewInfo);
		}

		// Create Current Sampled Image
		m_currentImage = nullptr;
		m_currentImageView = nullptr;

		{
			EuropaImageInfo info;
			info.width = m_amalthea.m_windowSize.x;
			info.height = m_amalthea.m_windowSize.y;
			info.initialLayout = EuropaImageLayout::General;
			info.type = EuropaImageType::Image2D;
			info.format = EuropaImageFormat::RGBA16F;
			info.usage = EuropaImageUsage(EuropaImageUsageStorage | EuropaImageUsageTransferSrc | EuropaImageUsageTransferDst);
			info.memoryUsage = EuropaMemoryUsage::GpuOnly;

			m_currentImage = m_amalthea.m_device->CreateImage(info);

			EuropaImageViewCreateInfo viewInfo;
			viewInfo.format = EuropaImageFormat::RGBA16F;
			viewInfo.image = m_currentImage;
			viewInfo.type = EuropaImageViewType::View2D;
			viewInfo.minArrayLayer = 0;
			viewInfo.minMipLevel = 0;
			viewInfo.numArrayLayers = 1;
			viewInfo.numMipLevels = 1;

			m_currentImageView = m_amalthea.m_device->CreateImageView(viewInfo);

			EuropaBufferInfo cpuBufferInfo;
			cpuBufferInfo.exclusive = true;
			cpuBufferInfo.size = uint32(m_amalthea.m_windowSize.x * m_amalthea.m_windowSize.y * sizeof(glm::u16vec4));
			cpuBufferInfo.usage = EuropaBufferUsage(EuropaBufferUsageTransferDst);
			cpuBufferInfo.memoryUsage = EuropaMemoryUsage::Gpu2Cpu;

			for (auto f : m_amalthea.m_frames)
			{
				m_currentImageCpuBuffers.push_back(m_amalthea.m_device->CreateBuffer(cpuBufferInfo));
			}
		}

		// Create Ray Stack
		EuropaBufferInfo rayStackInfo;
		rayStackInfo.exclusive = true;
		rayStackInfo.size = uint32(m_amalthea.m_windowSize.x * m_amalthea.m_windowSize.y * 5 * sizeof(RayStack));
		rayStackInfo.usage = EuropaBufferUsage(EuropaBufferUsageStorage | EuropaBufferUsageTransferSrc);
		rayStackInfo.memoryUsage = EuropaMemoryUsage::GpuOnly;
		m_rayStackBuffer = m_amalthea.m_device->CreateBuffer(rayStackInfo);

		// Create Job Buffer
		EuropaBufferInfo jobBufferInfo;
		jobBufferInfo.exclusive = true;
		jobBufferInfo.size = uint32(amalthea->m_windowSize.x * amalthea->m_windowSize.y * sizeof(RayJob));
		jobBufferInfo.usage = EuropaBufferUsage(EuropaBufferUsageStorage | EuropaBufferUsageTransferSrc);
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
		if (!sceneLoaded)
		{
			EuropaClearValue clearValue[2];
			clearValue[0].color = glm::vec4(0.0, 0.0, 0.0, 1.0);
			clearValue[1].depthStencil = glm::vec2(1.0, 0.0);
			ctx.cmdlist->BeginRenderpass(m_mainRenderPass, m_frameBuffers[ctx.frameIndex], glm::ivec2(0), glm::uvec2(amalthea->m_windowSize), 2, clearValue);
			ctx.cmdlist->EndRenderpass();

			ImGui::SetNextWindowPos(ImVec2(15, 15));
			ImGui::Begin("", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoMove);
			ImGui::Text("Loading scene %s", m_sceneFile.c_str());
			ImGui::BufferingBar("Progress", m_bvhBuildProgress, ImVec2(250, 6), ImU32(0xFF202020), ImU32(0xFF2080A0));
			ImGui::End();

			return;
		}

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

		constants->viewMtx = glm::lookAt(glm::vec3(cos(m_orbitAngle) * m_orbitRadius, m_orbitHeight, sin(m_orbitAngle) * m_orbitRadius) + m_focusCenter, m_focusCenter, glm::vec3(0.0, 1.0, 0.0));
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
		
		constants->ambientRadiance = m_ambientRadiance;

		constantsHandle.Unmap();

		m_descSets[ctx.frameIndex]->SetUniformBufferDynamic(constantsHandle.buffer, 0, constantsHandle.offset + m_constantsSize, 0, 0);
		if (!m_visualize)
		{
			m_descSets[ctx.frameIndex]->SetStorage(m_lightsBuffer, 0, uint32(lights.size() * sizeof(Light)), 1, 0);
			m_descSets[ctx.frameIndex]->SetStorage(m_vertexBuffer, 0, uint32(bvhVisStartVertex * sizeof(VertexAux)), 2, 0);
			m_descSets[ctx.frameIndex]->SetBufferViewUniform(m_indexBufferView, 3, 0);
			m_descSets[ctx.frameIndex]->SetStorage(m_blueNoiseBuffer, 0, sizeof(_blueNoise), 4, 0);
			
			m_descSets[ctx.frameIndex]->SetImageViewStorage(m_currentImageView, EuropaImageLayout::General, 5, 0);
			m_descSets[ctx.frameIndex]->SetImageViewStorage(m_accumulationImageViews, EuropaImageLayout::General, 6, 0);
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
				m_accumulationImages,
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

		if (m_dumpData)
		{
			ctx.cmdlist->Barrier(
				m_currentImage,
				EuropaAccessNone, EuropaAccessNone, EuropaImageLayout::General, EuropaImageLayout::TransferSrc,
				EuropaPipelineStageBottomOfPipe, EuropaPipelineStageTransfer
			);

			uint32 size = uint32((m_amalthea.m_windowSize.x * m_amalthea.m_windowSize.y) * sizeof(glm::u16vec4));
			WriteBuffer(m_currentImageCpuBuffers[ctx.frameIndex], m_frameIndex, size);
			ctx.cmdlist->CopyImageToBuffer(
				m_currentImageCpuBuffers[ctx.frameIndex], m_currentImage, EuropaImageLayout::TransferSrc,
				0, m_amalthea.m_windowSize.x, m_amalthea.m_windowSize.y,
				glm::uvec3(0), glm::uvec3(m_amalthea.m_windowSize.x, m_amalthea.m_windowSize.y, 1), 0
			);

			ctx.cmdlist->Barrier(
				m_currentImage,
				EuropaAccessNone, EuropaAccessNone, EuropaImageLayout::TransferSrc, EuropaImageLayout::General,
				EuropaPipelineStageTransfer, EuropaPipelineStageTopOfPipe
			);
		}

		m_fps = m_fps * 0.7f + 0.3f * glm::clamp(1.0f / deltaTime, m_fps - 10.0f, m_fps + 10.0f);
		m_frameTimeLog.AddPoint(time, deltaTime * 1000.0f);
		m_frameRateLog.AddPoint(time, m_fps);
		m_frameCount++;

		if (ImGui::Begin("Debug Info"))
		{
			ImGui::LabelText("", "CPU: %f ms", deltaTime * 1000.0);
			ImGui::LabelText("", "FPS: %f", m_fps);

			if (ImGui::DragFloat3("Center", &m_focusCenter.x)) clear = true;
			
			if (ImGui::SliderInt("Max Depth", (int*)&m_maxDepth, 1, 5)) clear = true;

			ImGui::Checkbox("Ray Sorting", &m_raySort);
			ImGui::SameLine();
			ImGui::Checkbox("Visualization", &m_visualize);
			ImGui::SameLine();
			if (ImGui::Button("Reset Image")) clear = true;

			ImGui::Checkbox("Dump Data", &m_dumpData);

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

		if (ImGui::Begin("Scene"))
		{
			const std::string scenes[] = { 
				"../Models/CBbunny.ply",
				"../Models/CBdragon.ply",
				"../Models/CBmonkey.ply",
				"../Models/minecraft.ply",
				"../Models/cornellBox.ply",
				"../Models/sponza.ply",
				"../Models/conference.ply",
				"../Models/livingRoom.ply",
				"../Models/SanMiguel.ply"
			};

			static std::string current_item = m_sceneFile;

			if (ImGui::BeginCombo("##combo", m_sceneFile.c_str())) // The second parameter is the label previewed before opening the combo.
			{
				for (int n = 0; n < IM_ARRAYSIZE(scenes); n++)
				{
					bool is_selected = (current_item.compare(scenes[n]) == 0); // You can store your selection however you want, outside or inside your objects
					if (ImGui::Selectable(scenes[n].c_str(), is_selected))
					{
						m_sceneFile = scenes[n];
						if (is_selected)
							ImGui::SetItemDefaultFocus();
						else
							ReloadScene();
					}
				}
				ImGui::EndCombo();
			}

			ImGui::Separator();

			ImGui::DragFloat3("Position", &lights[0].pos.x);
			ImGui::DragFloat3("Radiance", &lights[0].radiance.r, 0.5, 0.0);
			ImGui::DragFloat3("Ambient Radiance", &m_ambientRadiance.x, 0.5, 0.0);

			if (ImGui::Button("Update"))
			{
				clear = true;
				UpdateLights();
			}
		}
		ImGui::End();

		if (clear)
		{
			ctx.cmdlist->ClearImage(m_accumulationImages, EuropaImageLayout::General, glm::vec4(0.0));
		}
	};

	~TestApp()
	{
		while (!writeBuffers.empty())
		{
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
		}

		terminate = true;

		writeJobAvailable.notify_all();

		for (auto& w : workers)
		{
			w.join();
		}
	}

	TestApp(GanymedeECS& ecs, Europa& e, IoSurface::Ref s)
		: m_amalthea(ecs, e, s)
	{
		ecs.RegisterHandler(m_amalthea.m_events.OnCreateDevice, &f_onCreateDevice);
		ecs.RegisterHandler(m_amalthea.m_events.OnDestroyDevice, &f_onDestroyDevice);
		ecs.RegisterHandler(m_amalthea.m_events.OnCreateSwapChain, &f_onCreateSwapChain);
		ecs.RegisterHandler(m_amalthea.m_events.OnDestroySwapChain, &f_onDestroySwapChain);
		ecs.RegisterHandler(m_amalthea.m_events.OnRender, &f_onRender);

		terminate = false;
		for (int i = 0; i < std::thread::hardware_concurrency(); i++)
		{
			workers.push_back(std::thread([&] { WriteWorker(); }));
		}

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