#include <algorithm>
#include <array>
#include <chrono>
#include <glm/ext/matrix_transform.hpp>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#include <iterator>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <thread>
#include <vector>
#include <vulkan/vulkan.h>

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <fstream>

#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include <string.h>

#include <vulkan/vulkan_core.h>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define PRINT(x) std::cout << "(" << __LINE__ << ") " << x << '\n';
#define LOG(x)	 std::cout << "(" << __LINE__ << ") " << #x << ": " << x << '\n'

// Vizium

typedef int32_t	 i32;
typedef int64_t	 i64;
typedef uint32_t ui32;
typedef uint64_t ui64;

static std::vector<char> readFile(const std::string &filename)
{
	std::ifstream file(filename, std::ios::ate | std::ios::binary);

	if (!file.is_open())
	{
		throw std::runtime_error("failed to open file!");
	}

	size_t			  fileSize = (size_t)file.tellg();
	std::vector<char> buffer(fileSize);

	file.seekg(0);
	file.read(buffer.data(), fileSize);

	file.close();

	return buffer;
}

class Command
{
	public:
		VkCommandBuffer buffer;
		VkQueue			queue;
		VkDevice		device;
		VkCommandPool	pool;

		void begin()
		{
			VkCommandBufferBeginInfo beginInfo{};
			beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

			vkBeginCommandBuffer(buffer, &beginInfo);
		}

		void end(VkFence fence = nullptr, VkSemaphore *waitSemaphores = nullptr, ui32 waitSemaphoreCount = 0,
				 VkSemaphore *signalSemaphores = nullptr, ui32 signalSemaphoresCount = 0,
				 VkPipelineStageFlags *waitStages = nullptr)
		{
			vkEndCommandBuffer(buffer);

			VkSubmitInfo submitInfo{};
			submitInfo.sType				= VK_STRUCTURE_TYPE_SUBMIT_INFO;
			submitInfo.commandBufferCount	= 1;
			submitInfo.pCommandBuffers		= &buffer;
			submitInfo.pWaitSemaphores		= waitSemaphores;
			submitInfo.waitSemaphoreCount	= waitSemaphoreCount;
			submitInfo.pSignalSemaphores	= signalSemaphores;
			submitInfo.signalSemaphoreCount = signalSemaphoresCount;
			submitInfo.pWaitDstStageMask	= waitStages;

			vkQueueSubmit(queue, 1, &submitInfo, fence);
		}

		void reset()
		{
			vkResetCommandBuffer(buffer, 0);
		}

		void waitQueue()
		{
			vkQueueWaitIdle(queue);
		}
};

class CommandPool
{
	public:
		VkCommandPool commandPool;
		VkQueue		  queue;
		VkDevice	  device;

		Command create()
		{
			Command command;

			command.queue  = queue;
			command.device = device;
			command.pool   = commandPool;

			VkCommandBufferAllocateInfo allocInfo{};
			allocInfo.sType				 = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
			allocInfo.commandPool		 = commandPool;
			allocInfo.level				 = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
			allocInfo.commandBufferCount = 1;

			if (vkAllocateCommandBuffers(device, &allocInfo, &command.buffer) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to allocate command buffers!");
			}

			return command;
		}

		void destroy(Command command)
		{
			vkQueueWaitIdle(queue);

			vkFreeCommandBuffers(device, commandPool, 1, &command.buffer);
		}
};

class Queue
{
	public:
		VkQueue	 queue;
		uint32_t index;
		VkDevice device;

		enum Type
		{
			GRAPHICS,
			COMPUTE,
			TRANSFER,
		};

		CommandPool createCommandPool()
		{
			CommandPool pool;

			pool.device = device;
			pool.queue	= queue;

			VkCommandPoolCreateInfo poolInfo{};
			poolInfo.sType			  = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
			poolInfo.flags			  = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
			poolInfo.queueFamilyIndex = index;

			if (vkCreateCommandPool(device, &poolInfo, nullptr, &pool.commandPool) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to create command pool!");
			}

			return pool;
		}
};

class Map
{
	public:
		struct Data
		{
				VmaAllocation allocation;
				VmaAllocator  allocator;
				void		 *map;
		} data;

		Map(Data data)
		{
			this->data = data;
		};

		void write(void *test, int size)
		{
			memcpy(data.map, test, size);
		}

		~Map()
		{
			vmaUnmapMemory(data.allocator, data.allocation);
		}
};

class Buffer
{
	public:
		VkBuffer	  buffer;
		VmaAllocation allocation;
		int			  size;
		i32			  elementCount;

		VmaAllocator *allocator;

		void map(void **map)
		{
			vmaMapMemory(*allocator, allocation, map);
		}

		Map mapTest()
		{
			void *map;
			this->map(&map);

			return Map::Data{allocation, *allocator, map};
		}

		void unmap()
		{
			vmaUnmapMemory(*allocator, allocation);
		}

		void singleCopy(void *data)
		{
			Map map = mapTest();
			map.write(data, size);
		}

		void destroy()
		{
			vmaDestroyBuffer(*allocator, buffer, allocation);
		}
};

class Sampler
{
	public:
		VkSampler sampler;
};

class Image
{
	public:
		VkImage		image;
		VkImageView view;

		int width;
		int height;

		VmaAllocation allocation;
		VmaAllocator *allocator;

		VkImageLayout currentLayout;

		void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout newLayout, CommandPool commandPool,
								   VkDevice device, VkQueue graphicsQueue)
		{
			Command commandBuffer = commandPool.create();
			commandBuffer.begin();
			VkImageMemoryBarrier barrier{};
			barrier.sType	  = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			barrier.oldLayout = currentLayout;
			barrier.newLayout = newLayout;

			barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

			barrier.image							= image;
			barrier.subresourceRange.aspectMask		= VK_IMAGE_ASPECT_COLOR_BIT;
			barrier.subresourceRange.baseMipLevel	= 0;
			barrier.subresourceRange.levelCount		= 1;
			barrier.subresourceRange.baseArrayLayer = 0;
			barrier.subresourceRange.layerCount		= 1;

			VkPipelineStageFlags sourceStage;
			VkPipelineStageFlags destinationStage;

			if (currentLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
			{
				barrier.srcAccessMask = 0;
				barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

				sourceStage		 = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
				destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
			}
			else if (currentLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
					 newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
			{
				barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
				barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

				sourceStage		 = VK_PIPELINE_STAGE_TRANSFER_BIT;
				destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
			}
			else
			{
				throw std::invalid_argument("unsupported layout transition!");
			}

			vkCmdPipelineBarrier(commandBuffer.buffer, sourceStage, destinationStage, 0, 0, nullptr, 0, nullptr, 1,
								 &barrier);

			commandBuffer.end(nullptr);
			commandPool.destroy(commandBuffer);

			currentLayout = newLayout;
		}

		void copyFromBuffer(Buffer buffer, CommandPool commandPool, VkDevice device, VkQueue graphicsQueue)
		{
			transitionImageLayout(image, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, commandPool,
								  device, graphicsQueue);

			Command command = commandPool.create();
			command.begin();

			VkBufferImageCopy region{};
			region.bufferOffset		 = 0;
			region.bufferRowLength	 = 0;
			region.bufferImageHeight = 0;

			region.imageSubresource.aspectMask	   = VK_IMAGE_ASPECT_COLOR_BIT;
			region.imageSubresource.mipLevel	   = 0;
			region.imageSubresource.baseArrayLayer = 0;
			region.imageSubresource.layerCount	   = 1;

			region.imageOffset = {0, 0, 0};
			region.imageExtent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1};

			vkCmdCopyBufferToImage(command.buffer, buffer.buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
								   &region);

			command.end();
			commandPool.destroy(command);

			transitionImageLayout(image, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, commandPool,
								  device, graphicsQueue);
		}

		void destroy(VkDevice device)
		{
			vkDestroyImageView(device, view, nullptr);
			vmaDestroyImage(*allocator, image, allocation);
		}
};

class Allocator
{
	public:
		VmaAllocator	 vmaAllocator;
		VkDevice		 device;
		VkPhysicalDevice physicalDevice;

		void setup(VkPhysicalDevice physicalDevice, VkDevice device, VkInstance instance)
		{
			VmaAllocatorCreateInfo allocatorInfo = {};
			allocatorInfo.vulkanApiVersion		 = VK_API_VERSION_1_3;
			allocatorInfo.physicalDevice		 = physicalDevice;
			allocatorInfo.device				 = device;
			allocatorInfo.instance				 = instance;

			this->device		 = device;
			this->physicalDevice = physicalDevice;

			if (vmaCreateAllocator(&allocatorInfo, &vmaAllocator) != VK_SUCCESS)
			{
				throw std::runtime_error("Failed to create VMA allocator");
			}
		}

		Buffer allocate(int size, unsigned int usage,
						VmaAllocationCreateInfo allocatorInfo = {
							.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
							.usage = VMA_MEMORY_USAGE_CPU_ONLY})
		{
			Buffer buffer;
			buffer.allocator = &vmaAllocator;
			buffer.size		 = size;

			VkBufferCreateInfo bufferInfo{};
			bufferInfo.sType	   = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
			bufferInfo.size		   = size;
			bufferInfo.usage	   = usage;
			bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
			bufferInfo.flags	   = 0;
			bufferInfo.pNext	   = nullptr;

			vmaCreateBuffer(vmaAllocator, &bufferInfo, &allocatorInfo, &buffer.buffer, &buffer.allocation, nullptr);

			return buffer;
		}

		Image createImage(
			uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage,
			VmaAllocationCreateInfo allocatorInfo = {.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
													 .usage = VMA_MEMORY_USAGE_CPU_ONLY},
			VkImageAspectFlags		aspect		  = VK_IMAGE_ASPECT_COLOR_BIT)
		{
			Image image;
			image.allocator		= &vmaAllocator;
			image.width			= width;
			image.height		= height;
			image.currentLayout = VK_IMAGE_LAYOUT_UNDEFINED;

			VkImageCreateInfo imageInfo{};
			imageInfo.sType			= VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
			imageInfo.imageType		= VK_IMAGE_TYPE_2D;
			imageInfo.extent.width	= width;
			imageInfo.extent.height = height;
			imageInfo.extent.depth	= 1;
			imageInfo.mipLevels		= 1;
			imageInfo.arrayLayers	= 1;
			imageInfo.format		= format;
			imageInfo.tiling		= tiling;
			imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			imageInfo.usage			= usage;
			imageInfo.samples		= VK_SAMPLE_COUNT_1_BIT;
			imageInfo.sharingMode	= VK_SHARING_MODE_EXCLUSIVE;

			if (vmaCreateImage(vmaAllocator, &imageInfo, &allocatorInfo, &image.image, &image.allocation, nullptr) !=
				VK_SUCCESS)
			{
				throw std::runtime_error("failed to create image!");
			}

			VkImageViewCreateInfo viewInfo{};
			viewInfo.sType							 = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			viewInfo.image							 = image.image;
			viewInfo.viewType						 = VK_IMAGE_VIEW_TYPE_2D;
			viewInfo.format							 = format;
			viewInfo.subresourceRange.aspectMask	 = aspect;
			viewInfo.subresourceRange.baseMipLevel	 = 0;
			viewInfo.subresourceRange.levelCount	 = 1;
			viewInfo.subresourceRange.baseArrayLayer = 0;
			viewInfo.subresourceRange.layerCount	 = 1;

			if (vkCreateImageView(device, &viewInfo, nullptr, &image.view) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to create texture image view!");
			}

			return image;
		}

		void free(Buffer buffer)
		{
			vmaDestroyBuffer(vmaAllocator, buffer.buffer, buffer.allocation);
		}

		void destroy()
		{
			vmaDestroyAllocator(vmaAllocator);
		}
};

class DescriptorLayout
{
	public:
		VkDescriptorSetLayout layout;
		int					  binding;

		enum Type
		{
			UNIFORM_BUFFER		   = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			COMBINED_IMAGE_SAMPLER = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			STORAGE_BUFFER		   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		};

		Type type;

		enum Stage
		{
			VERTEX	 = VK_SHADER_STAGE_VERTEX_BIT,
			FRAGMENT = VK_SHADER_STAGE_FRAGMENT_BIT,
			COMPUTE	 = VK_SHADER_STAGE_COMPUTE_BIT,
		};
};

class Descriptor
{
	public:
		VkDescriptorSet descriptorSets;

		VkWriteDescriptorSet setupMeta(VkDescriptorPool descriptorPool, VkDescriptorSetLayout descriptorSetLayout,
									   VkDevice device, int binding)
		{
			VkDescriptorSetLayout		layouts = descriptorSetLayout;
			VkDescriptorSetAllocateInfo allocInfo{};
			allocInfo.sType				 = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
			allocInfo.descriptorPool	 = descriptorPool;
			allocInfo.descriptorSetCount = 1;
			allocInfo.pSetLayouts		 = &layouts;

			if (vkAllocateDescriptorSets(device, &allocInfo, &descriptorSets) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to allocate descriptor sets!");
			}

			VkWriteDescriptorSet descriptorWrite;
			descriptorWrite.sType			= VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrite.dstSet			= descriptorSets;
			descriptorWrite.dstBinding		= binding;
			descriptorWrite.dstArrayElement = 0;
			descriptorWrite.descriptorCount = 1;
			descriptorWrite.pNext			= nullptr;

			return descriptorWrite;
		}
};

class Pipeline
{
	public:
		VkPipeline		 pipeline;
		VkPipelineLayout layout;
};

class DescriptorPool
{
	public:
		VkDescriptorPool descriptorPool;
		VkDevice		 device;

		Descriptor createDescriptor(DescriptorLayout descriptorSetLayout, Buffer *buffer, Image *image,
									Sampler *sampler)
		{
			Descriptor descriptor;

			VkDescriptorSetAllocateInfo allocInfo{};
			allocInfo.sType				 = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
			allocInfo.descriptorPool	 = descriptorPool;
			allocInfo.descriptorSetCount = 1;
			allocInfo.pSetLayouts		 = &descriptorSetLayout.layout;

			if (vkAllocateDescriptorSets(device, &allocInfo, &descriptor.descriptorSets) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to allocate descriptor sets!");
			}

			VkWriteDescriptorSet descriptorWrite;
			descriptorWrite.sType			= VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrite.dstSet			= descriptor.descriptorSets;
			descriptorWrite.dstBinding		= descriptorSetLayout.binding;
			descriptorWrite.dstArrayElement = 0;
			descriptorWrite.descriptorCount = 1;
			descriptorWrite.pNext			= nullptr;

			VkDescriptorBufferInfo bufferInfo;
			if (buffer != nullptr)
			{
				bufferInfo.buffer = buffer->buffer;
				bufferInfo.range  = buffer->size;
				bufferInfo.offset = 0;
			}

			VkDescriptorImageInfo imageInfo;
			if (sampler != nullptr)
			{
				imageInfo.sampler = sampler->sampler;
			}

			if (image != nullptr)
			{
				imageInfo.imageView	  = image->view;
				imageInfo.imageLayout = image->currentLayout;
			}

			descriptorWrite.descriptorType	 = (VkDescriptorType)descriptorSetLayout.type;
			descriptorWrite.pBufferInfo		 = buffer != nullptr ? &bufferInfo : nullptr;
			descriptorWrite.pImageInfo		 = image != nullptr ? &imageInfo : nullptr;
			descriptorWrite.pTexelBufferView = nullptr;

			vkUpdateDescriptorSets(device, 1, &descriptorWrite, 0, nullptr);

			return descriptor;
		}

		void reset()
		{
			vkResetDescriptorPool(device, descriptorPool, 0);
		}
};

class Window
{
	public:
		GLFWwindow				  *window;
		bool					   framebufferResized = false;
		Queue					   graphicsQueue;
		CommandPool				   graphicsPool;
		Command					   graphicsBuffer;
		VkSurfaceKHR			   surface;
		VkSwapchainKHR			   swapChain;
		VkFormat				   swapChainImageFormat;
		VkExtent2D				   swapChainExtent;
		std::vector<VkImage>	   swapChainImages;
		std::vector<VkImageView>   swapChainImageViews;
		std::vector<VkFramebuffer> swapChainFramebuffers;
		VkRenderPass			   renderPass;
		std::vector<uint32_t>	  *uniqueQueues;
		VkSemaphore				   imageAvailableSemaphores;
		VkSemaphore				   renderFinishedSemaphores;
		VkFence					   inFlightFences;
		uint32_t				   imageIndex;
		Image					   depthImage;

		VkDevice		 device;
		VkPhysicalDevice physicalDevice;

		bool		   imguiEnabled = false;
		DescriptorPool imguiDescriptorPool;

		std::chrono::time_point<std::chrono::system_clock> lastFrameTime;

		void waitTillInterval(int milliseconds)
		{
			std::this_thread::sleep_until(lastFrameTime + std::chrono::milliseconds(milliseconds));
		}

		void createSyncObjects()
		{
			VkSemaphoreCreateInfo semaphoreInfo{};
			semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

			VkFenceCreateInfo fenceInfo{};
			fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
			fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

			if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores) != VK_SUCCESS ||
				vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores) != VK_SUCCESS ||
				vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to create semaphores!");
			}
		}
		void createFrameBuffers()
		{
			swapChainFramebuffers.resize(swapChainImageViews.size());

			for (size_t i = 0; i < swapChainImageViews.size(); i++)
			{
				std::array<VkImageView, 2> attachments = {swapChainImageViews[i], depthImage.view};

				VkFramebufferCreateInfo framebufferInfo{};
				framebufferInfo.sType			= VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
				framebufferInfo.renderPass		= renderPass;
				framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
				framebufferInfo.pAttachments	= attachments.data();
				framebufferInfo.width			= swapChainExtent.width;
				framebufferInfo.height			= swapChainExtent.height;
				framebufferInfo.layers			= 1;

				if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS)
				{
					throw std::runtime_error("failed to create framebuffer!");
				}
			}
		}

		void draw(int vertices, std::vector<Descriptor> descriptors, Pipeline &pipeline)
		{
			vkCmdBindPipeline(graphicsBuffer.buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.pipeline);

			VkViewport viewport{};
			viewport.x		  = 0.0f;
			viewport.y		  = 0.0f;
			viewport.width	  = static_cast<float>(swapChainExtent.width);
			viewport.height	  = static_cast<float>(swapChainExtent.height);
			viewport.minDepth = 0.0f;
			viewport.maxDepth = 1.0f;
			vkCmdSetViewport(graphicsBuffer.buffer, 0, 1, &viewport);

			VkRect2D scissor{};
			scissor.offset = {0, 0};
			scissor.extent = swapChainExtent;
			vkCmdSetScissor(graphicsBuffer.buffer, 0, 1, &scissor);

			std::vector<VkDescriptorSet> sets(descriptors.size());
			int							 i = 0;
			for (auto &s : sets)
			{
				s = descriptors[i].descriptorSets;
				i++;
			}

			vkCmdBindDescriptorSets(graphicsBuffer.buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.layout, 0,
									sets.size(), sets.data(), 0, nullptr);

			vkCmdDraw(graphicsBuffer.buffer, vertices, 1, 0, 0);
		}

		void endDraw()
		{
			if (imguiEnabled)
			{
				ImGui::Render();
				ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), graphicsBuffer.buffer);
				ImGui_ImplVulkan_NewFrame();
				ImGui_ImplGlfw_NewFrame();
				ImGui::NewFrame();
			}

			vkCmdEndRenderPass(graphicsBuffer.buffer);

			VkSemaphore			 waitSemaphores[]	= {imageAvailableSemaphores};
			VkPipelineStageFlags waitStages[]		= {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
			VkSemaphore			 signalSemaphores[] = {renderFinishedSemaphores};

			graphicsBuffer.end(inFlightFences, waitSemaphores, 1, signalSemaphores, 1, waitStages);

			VkPresentInfoKHR presentInfo{};
			presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

			presentInfo.waitSemaphoreCount = 1;
			presentInfo.pWaitSemaphores	   = signalSemaphores;

			VkSwapchainKHR swapChains[] = {swapChain};
			presentInfo.swapchainCount	= 1;
			presentInfo.pSwapchains		= swapChains;
			presentInfo.pImageIndices	= &imageIndex;

			presentInfo.pResults = nullptr; // optional

		waitTillInterval(10);
			VkResult result = vkQueuePresentKHR(graphicsQueue.queue, &presentInfo);

			if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized)
			{
				framebufferResized = false;
				recreateSwapChain();
			}
			else if (result != VK_SUCCESS)
			{
				throw std::runtime_error("failed to present swap chain image!");
			}

			lastFrameTime = std::chrono::system_clock::now();
		}

		void startDraw()
		{
			vkWaitForFences(device, 1, &inFlightFences, VK_TRUE, UINT64_MAX);

			VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphores,
													VK_NULL_HANDLE, &imageIndex);

			if (result == VK_ERROR_OUT_OF_DATE_KHR)
			{
				recreateSwapChain();

				return;
			}
			else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
			{
				throw std::runtime_error("failed to acquire swap chain image!");
			}

			vkResetFences(device, 1, &inFlightFences);

			vkResetCommandBuffer(graphicsBuffer.buffer, 0);

			static int frame = 0;
			frame++;

			graphicsBuffer.begin();

			VkRenderPassBeginInfo renderPassInfo{};
			renderPassInfo.sType	   = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
			renderPassInfo.renderPass  = renderPass;
			renderPassInfo.framebuffer = swapChainFramebuffers[imageIndex];

			renderPassInfo.renderArea.offset = {0, 0};
			renderPassInfo.renderArea.extent = swapChainExtent;

			std::array<VkClearValue, 2> clearValues{};
			clearValues[0].color		= {{0.0f, 0.0f, 0.0f, 1.0f}};
			clearValues[1].depthStencil = {1.0f, 0};

			renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
			renderPassInfo.pClearValues	   = clearValues.data();

			vkCmdBeginRenderPass(graphicsBuffer.buffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
		}

		void createRenderPass()
		{
			VkAttachmentDescription depthAttachment;
			depthAttachment.format		   = VK_FORMAT_D32_SFLOAT;
			depthAttachment.samples		   = VK_SAMPLE_COUNT_1_BIT;
			depthAttachment.loadOp		   = VK_ATTACHMENT_LOAD_OP_CLEAR;
			depthAttachment.storeOp		   = VK_ATTACHMENT_STORE_OP_DONT_CARE;
			depthAttachment.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
			depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
			depthAttachment.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
			depthAttachment.finalLayout	   = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

			VkAttachmentDescription colorAttachment{};
			colorAttachment.format	= swapChainImageFormat;
			colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;

			colorAttachment.loadOp	= VK_ATTACHMENT_LOAD_OP_CLEAR;
			colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

			colorAttachment.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
			colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

			colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			colorAttachment.finalLayout	  = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

			VkAttachmentReference colorAttachmentRef{};
			colorAttachmentRef.attachment = 0;
			colorAttachmentRef.layout	  = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

			VkAttachmentReference depthAttachmentRef{};
			depthAttachmentRef.attachment = 1;
			depthAttachmentRef.layout	  = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

			VkSubpassDescription subpass{};
			subpass.pipelineBindPoint		= VK_PIPELINE_BIND_POINT_GRAPHICS;
			subpass.colorAttachmentCount	= 1;
			subpass.pColorAttachments		= &colorAttachmentRef;
			subpass.pDepthStencilAttachment = &depthAttachmentRef;

			VkSubpassDependency dependency{};
			dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
			dependency.dstSubpass = 0;
			dependency.srcStageMask =
				VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
			dependency.srcAccessMask = 0;
			dependency.dstStageMask =
				VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
			dependency.dstAccessMask =
				VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

			std::array<VkAttachmentDescription, 2> attachments = {colorAttachment, depthAttachment};
			VkRenderPassCreateInfo				   renderPassInfo{};
			renderPassInfo.sType		   = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
			renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
			renderPassInfo.pAttachments	   = attachments.data();
			renderPassInfo.subpassCount	   = 1;
			renderPassInfo.pSubpasses	   = &subpass;
			renderPassInfo.dependencyCount = 1;
			renderPassInfo.pDependencies   = &dependency;

			if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to create render pass!");
			}
		}

		void cleanupSwapChain()
		{
			for (size_t i = 0; i < swapChainFramebuffers.size(); i++)
			{
				vkDestroyFramebuffer(device, swapChainFramebuffers[i], nullptr);
			}

			for (size_t i = 0; i < swapChainImageViews.size(); i++)
			{
				vkDestroyImageView(device, swapChainImageViews[i], nullptr);
			}

			vkDestroySwapchainKHR(device, swapChain, nullptr);
		}

		void recreateSwapChain()
		{
			int width = 0, height = 0;
			getFrameBufferSize(&width, &height);
			while (width == 0 || height == 0)
			{
				getFrameBufferSize(&width, &height);
				sleepTillEvent();
			}

			vkDeviceWaitIdle(device);

			cleanupSwapChain();

			createSwapChain();
			createImageViews();
			createFrameBuffers();
		}

		void createImageViews()
		{
			swapChainImageViews.resize(swapChainImages.size());
			for (size_t i = 0; i < swapChainImages.size(); i++)
			{
				VkImageViewCreateInfo createInfo{};
				createInfo.sType						   = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
				createInfo.image						   = swapChainImages[i];
				createInfo.viewType						   = VK_IMAGE_VIEW_TYPE_2D;
				createInfo.format						   = swapChainImageFormat;
				createInfo.components.r					   = VK_COMPONENT_SWIZZLE_IDENTITY;
				createInfo.components.g					   = VK_COMPONENT_SWIZZLE_IDENTITY;
				createInfo.components.b					   = VK_COMPONENT_SWIZZLE_IDENTITY;
				createInfo.components.a					   = VK_COMPONENT_SWIZZLE_IDENTITY;
				createInfo.subresourceRange.aspectMask	   = VK_IMAGE_ASPECT_COLOR_BIT;
				createInfo.subresourceRange.baseMipLevel   = 0;
				createInfo.subresourceRange.levelCount	   = 1;
				createInfo.subresourceRange.baseArrayLayer = 0;
				createInfo.subresourceRange.layerCount	   = 1;

				if (vkCreateImageView(device, &createInfo, nullptr, &swapChainImageViews[i]) != VK_SUCCESS)
				{
					throw std::runtime_error("failed to create image views!");
				}
			}
		}

		void setup(int width, int height, std::string name)
		{
			glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
			glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

			window = glfwCreateWindow(width, height, name.c_str(), nullptr, nullptr);

			glfwSetWindowUserPointer(window, this);
			glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
		}

		struct SwapChainSupportDetails
		{
				VkSurfaceCapabilitiesKHR		capabilities;
				std::vector<VkSurfaceFormatKHR> formats;
				std::vector<VkPresentModeKHR>	presentModes;
		};

		SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device)
		{
			SwapChainSupportDetails details;

			vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

			uint32_t formatCount;
			vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

			if (formatCount != 0)
			{
				details.formats.resize(formatCount);
				vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
			}

			uint32_t presentModeCount;
			vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

			if (presentModeCount != 0)
			{
				details.presentModes.resize(presentModeCount);
				vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount,
														  details.presentModes.data());
			}

			return details;
		}

		VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR &capabilities)
		{
			if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
			{
				return capabilities.currentExtent;
			}
			else
			{
				int width, height;
				getFrameBufferSize(&width, &height);

				VkExtent2D actualExtent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};

				actualExtent.width	= std::clamp(actualExtent.width, capabilities.minImageExtent.width,
												 capabilities.maxImageExtent.width);
				actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height,
												 capabilities.maxImageExtent.height);

				return actualExtent;
			}
		}

		VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR> &availableFormats)
		{
			for (const auto &availableFormat : availableFormats)
			{
				if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB &&
					availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
				{
					return availableFormat;
				}
			}

			return availableFormats[0];
		}

		VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR> &availablePresentModes)
		{
			for (const auto &availablePresentMode : availablePresentModes)
			{
				if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR)
				{
					return availablePresentMode;
				}
			}

			return VK_PRESENT_MODE_FIFO_KHR;
		}

		void createSwapChain()
		{
			SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

			VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
			VkPresentModeKHR   presentMode	 = chooseSwapPresentMode(swapChainSupport.presentModes);
			VkExtent2D		   extent		 = chooseSwapExtent(swapChainSupport.capabilities);

			uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;

			if (swapChainSupport.capabilities.maxImageCount > 0 &&
				imageCount > swapChainSupport.capabilities.maxImageCount)
			{
				imageCount = swapChainSupport.capabilities.maxImageCount;
			}

			VkSwapchainCreateInfoKHR createInfo{};
			createInfo.sType   = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
			createInfo.surface = surface;

			createInfo.minImageCount	= imageCount;
			createInfo.imageFormat		= surfaceFormat.format;
			createInfo.imageColorSpace	= surfaceFormat.colorSpace;
			createInfo.imageExtent		= extent;
			createInfo.imageArrayLayers = 1;
			createInfo.imageUsage		= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

			if (uniqueQueues->size() > 1)
			{
				createInfo.imageSharingMode		 = VK_SHARING_MODE_CONCURRENT;
				createInfo.queueFamilyIndexCount = uniqueQueues->size();
				createInfo.pQueueFamilyIndices	 = uniqueQueues->data();
			}
			else
			{
				createInfo.imageSharingMode		 = VK_SHARING_MODE_EXCLUSIVE;
				createInfo.queueFamilyIndexCount = 0;		// Optional
				createInfo.pQueueFamilyIndices	 = nullptr; // Optional
			}

			createInfo.preTransform	  = swapChainSupport.capabilities.currentTransform;
			createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;

			createInfo.presentMode = presentMode;
			createInfo.clipped	   = VK_TRUE;

			createInfo.oldSwapchain = VK_NULL_HANDLE;

			if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to create swap chain!");
			}

			vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
			swapChainImages.resize(imageCount);
			vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());

			LOG(imageCount);

			swapChainImageFormat = surfaceFormat.format;
			swapChainExtent		 = extent;
		}

		void getFrameBufferSize(int *width, int *height)
		{
			glfwGetFramebufferSize(window, width, height);
		}

		void sleepTillEvent()
		{
			glfwWaitEvents();
		}

		VkSurfaceKHR createWindowSurface(VkInstance instance)
		{
			VkSurfaceKHR surface;

			if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to create window surface!");
			}

			return surface;
		}

		bool shouldClose()
		{
			return glfwWindowShouldClose(window);
		}

		static void framebufferResizeCallback(GLFWwindow *window, int width, int height)
		{
			((Window *)glfwGetWindowUserPointer(window))->framebufferResized = true;
		}

		void destroy()
		{
			glfwDestroyWindow(window);
		}
};

class Dispatcher
{
	public:
		Queue		computeQueue;
		CommandPool computePool;
		Command		computeCommand;

		void startDispatch()
		{
			computeCommand.begin();
		}

		void dispatch(Pipeline pipeline, std::vector<Descriptor> descriptor, int x, int y, int z)
		{
			vkCmdBindPipeline(computeCommand.buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);

			std::vector<VkDescriptorSet> sets(descriptor.size());
			int							 i = 0;
			for (auto &s : sets)
			{
				s = descriptor[i].descriptorSets;
				i++;
			}

			vkCmdBindDescriptorSets(computeCommand.buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.layout, 0,
									sets.size(), sets.data(), 0, 0);

			vkCmdDispatch(computeCommand.buffer, x, y, z);
		}
		void endDispatch()
		{
			computeCommand.end(nullptr);
			computeCommand.waitQueue();
		}
};

typedef std::map<Queue::Type, uint32_t> QueueMap;

class Instance
{
	public:
		VkInstance				 instance;
		VkDebugUtilsMessengerEXT debugMessenger;
		VkPhysicalDevice		 physicalDevice = VK_NULL_HANDLE;
		VkDevice				 device;

		QueueMap			  queueMap;
		std::vector<uint32_t> uniqueQueues;
		Queue				  transferQueue;
		CommandPool			  transferPool;

		Allocator vmaAllocator;

		bool enableValidationLayers;

		const std::vector<const char *> validationLayers = {"VK_LAYER_KHRONOS_validation"};
		const std::vector<const char *> deviceExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

		Instance()
		{
		}

		Dispatcher createDispatcher()
		{
			Dispatcher dispatcher;
			dispatcher.computeQueue	  = this->createQueue(Queue::Type::COMPUTE);
			dispatcher.computePool	  = dispatcher.computeQueue.createCommandPool();
			dispatcher.computeCommand = dispatcher.computePool.create();

			return dispatcher;
		}

		Window createWindow(int width, int height, std::string name, bool enableImGUI = false)
		{
			Window window;

			window.depthImage =
				vmaAllocator.createImage(width, height, VK_FORMAT_D24_UNORM_S8_UINT, VK_IMAGE_TILING_OPTIMAL,
										 VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, {.usage = VMA_MEMORY_USAGE_AUTO},
										 VK_IMAGE_ASPECT_DEPTH_BIT);

			window.uniqueQueues	  = &uniqueQueues;
			window.device		  = device;
			window.physicalDevice = physicalDevice;
			window.setup(width, height, name);
			window.surface = window.createWindowSurface(instance);
			window.createSwapChain();
			window.createImageViews();
			window.createRenderPass();
			window.createFrameBuffers();
			window.createSyncObjects();
			window.graphicsQueue  = createQueue(Queue::Type::GRAPHICS);
			window.graphicsPool	  = window.graphicsQueue.createCommandPool();
			window.graphicsBuffer = window.graphicsPool.create();

			if (enableImGUI)
			{
				window.imguiEnabled		   = true;
				window.imguiDescriptorPool = createDescriptorPool(1000);

				ImGui::CreateContext();
				ImGuiIO &io = ImGui::GetIO();
				ImGui_ImplGlfw_InitForVulkan(window.window, true);
				ImGui_ImplVulkan_InitInfo init_info = {};
				init_info.Instance					= instance;
				init_info.PhysicalDevice			= physicalDevice;
				init_info.Device					= device;
				init_info.QueueFamily				= window.graphicsQueue.index;
				init_info.Queue						= window.graphicsQueue.queue;
				init_info.DescriptorPool			= window.imguiDescriptorPool.descriptorPool;
				init_info.Subpass					= 0;
				init_info.MinImageCount				= 2;
				init_info.ImageCount				= 2;
				init_info.MSAASamples				= VK_SAMPLE_COUNT_1_BIT;
				init_info.RenderPass				= window.renderPass;
				ImGui_ImplVulkan_Init(&init_info);
				ImGui_ImplVulkan_CreateFontsTexture();

				ImGui_ImplVulkan_NewFrame();
				ImGui_ImplGlfw_NewFrame();
				ImGui::NewFrame();
			}
			return window;
		}

		bool checkValidationLayerSupport()
		{
			uint32_t layerCount;
			vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

			std::vector<VkLayerProperties> availableLayers(layerCount);
			vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

			for (const char *layerName : validationLayers)
			{
				bool layerFound = false;

				for (const auto &layerProperties : availableLayers)
				{
					if (strcmp(layerName, layerProperties.layerName) == 0)
					{
						layerFound = true;
						break;
					}
				}

				if (!layerFound)
				{
					return false;
				}
			}

			return true;
		}

		std::vector<const char *> getRequiredExtensions()
		{
			uint32_t	 glfwExtensionCount = 0;
			const char **glfwExtensions;
			glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

			std::vector<const char *> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

			if (enableValidationLayers)
			{
				extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
			}

			return extensions;
		}

		void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT &createInfo)
		{
			createInfo				   = {};
			createInfo.sType		   = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
			createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | //
										 /*VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |	   //*/
										 VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | //
										 VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;	   //
			createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
									 VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
									 VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
			createInfo.pfnUserCallback = debugCallback;
			createInfo.pUserData	   = nullptr;
		}

		static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT		messageSeverity,
															VkDebugUtilsMessageTypeFlagsEXT				messageType,
															const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
															void									   *pUserData)
		{

			if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
			{
				// Message is important enough to show
			}

			std::cerr << "validation layer: " << pCallbackData->pMessage << "\n\n\n";

			return VK_FALSE;
		}

		void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger,
										   const VkAllocationCallbacks *pAllocator)
		{
			auto func =
				(PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
			if (func != nullptr)
			{
				func(instance, debugMessenger, pAllocator);
			}
		}

		VkResult CreateDebugUtilsMessengerEXT(VkInstance								instance,
											  const VkDebugUtilsMessengerCreateInfoEXT *pCreateInfo,
											  const VkAllocationCallbacks			   *pAllocator,
											  VkDebugUtilsMessengerEXT				   *pDebugMessenger)
		{
			auto func =
				(PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
			if (func != nullptr)
			{
				return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
			}
			else
			{
				return VK_ERROR_EXTENSION_NOT_PRESENT;
			}
		}

		void bootstrap(float debugging)
		{
			enableValidationLayers = debugging;

			if (enableValidationLayers)
			{
				std::cout << "debugging" << '\n';
			}
			if (enableValidationLayers && !checkValidationLayerSupport())
			{
				throw std::runtime_error("validation layers requested, but not available!");
			}

			VkApplicationInfo appInfo{};
			appInfo.sType			   = VK_STRUCTURE_TYPE_APPLICATION_INFO;
			appInfo.pApplicationName   = "Hello Triangle";
			appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
			appInfo.pEngineName		   = "No Engine";
			appInfo.engineVersion	   = VK_MAKE_VERSION(1, 0, 0);
			appInfo.apiVersion		   = VK_API_VERSION_1_3;

			VkInstanceCreateInfo createInfo{};
			createInfo.sType			= VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
			createInfo.pApplicationInfo = &appInfo;

			if (enableValidationLayers)
			{
				createInfo.enabledLayerCount   = static_cast<uint32_t>(validationLayers.size());
				createInfo.ppEnabledLayerNames = validationLayers.data();
			}
			else
			{
				createInfo.enabledLayerCount = 0;
			}

			auto extensions					   = getRequiredExtensions();
			createInfo.enabledExtensionCount   = static_cast<uint32_t>(extensions.size());
			createInfo.ppEnabledExtensionNames = extensions.data();

			for (auto s : extensions)
			{
				std::cout << s << '\n';
			}

			VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
			if (enableValidationLayers)
			{
				createInfo.enabledLayerCount   = static_cast<uint32_t>(validationLayers.size());
				createInfo.ppEnabledLayerNames = validationLayers.data();

				populateDebugMessengerCreateInfo(debugCreateInfo);
				createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT *)&debugCreateInfo;
			}
			else
			{
				createInfo.enabledLayerCount = 0;

				createInfo.pNext = nullptr;
			}

			if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to create instance!");
			}

			// SETUP VALIDATION

			if (enableValidationLayers)
			{

				VkDebugUtilsMessengerCreateInfoEXT createInfo{};
				populateDebugMessengerCreateInfo(createInfo);

				if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS)
				{
					throw std::runtime_error("failed to set up debug messenger!");
				}
			}

			// PICK PHYSICAL DEVICE

			uint32_t deviceCount = 0;

			vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

			if (deviceCount == 0)
			{
				throw std::runtime_error("failed to find GPUs with Vulkan support!");
			}

			std::vector<VkPhysicalDevice> devices(deviceCount);
			vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

			// Use an ordered map to automatically sort candidates by increasing score
			std::multimap<int, VkPhysicalDevice> candidates;

			for (const auto &device : devices)
			{
				int score = rateDeviceSuitability(device);
				candidates.insert(std::make_pair(score, device));
			}

			// Check if the best candidate is suitable at all
			if (candidates.rbegin()->first > 0)
			{
				physicalDevice = candidates.rbegin()->second;
				queueMap	   = findQueueFamilies(physicalDevice);
			}
			else
			{
				throw std::runtime_error("failed to find a suitable GPU!");
			}

			// PICK LOGICAL DEVICE

			{
				std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;

				std::set<uint32_t> unique;
				for (auto &key : queueMap)
				{
					unique.insert(key.second);
				}

				for (auto e : unique)
				{
					uniqueQueues.push_back(e);
				}

				float queuePriority = 1.0f;
				for (auto &uniqueQueueFamily : uniqueQueues)
				{
					VkDeviceQueueCreateInfo queueCreateInfo{};
					queueCreateInfo.sType			 = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
					queueCreateInfo.queueFamilyIndex = uniqueQueueFamily;
					queueCreateInfo.queueCount		 = 1;
					queueCreateInfo.pQueuePriorities = &queuePriority;
					queueCreateInfos.push_back(queueCreateInfo);
				}

				VkPhysicalDeviceFeatures deviceFeatures{};

				deviceFeatures.samplerAnisotropy = VK_TRUE;

				VkDeviceCreateInfo createInfo{};
				createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

				createInfo.pQueueCreateInfos	= queueCreateInfos.data();
				createInfo.queueCreateInfoCount = queueCreateInfos.size();

				createInfo.pEnabledFeatures = &deviceFeatures;

				createInfo.enabledExtensionCount   = static_cast<uint32_t>(deviceExtensions.size());
				createInfo.ppEnabledExtensionNames = deviceExtensions.data();

				if (enableValidationLayers)
				{
					createInfo.enabledLayerCount   = static_cast<uint32_t>(validationLayers.size());
					createInfo.ppEnabledLayerNames = validationLayers.data();
				}
				else
				{
					createInfo.enabledLayerCount = 0;
				}

				if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS)
				{
					throw std::runtime_error("failed to create logical device!");
				}
			}

			transferQueue = this->createQueue(Queue::Type::TRANSFER);
			transferPool  = transferQueue.createCommandPool();
			vmaAllocator.setup(physicalDevice, device, instance);
		}

		bool checkDeviceExtensionSupport(VkPhysicalDevice device)
		{
			uint32_t extensionCount;
			vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

			std::vector<VkExtensionProperties> availableExtensions(extensionCount);
			vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

			std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

			for (const auto &extension : availableExtensions)
			{
				requiredExtensions.erase(extension.extensionName);
			}

			return requiredExtensions.empty();
		}

		int rateDeviceSuitability(VkPhysicalDevice device)
		{
			VkPhysicalDeviceProperties deviceProperties;
			vkGetPhysicalDeviceProperties(device, &deviceProperties);

			VkPhysicalDeviceFeatures deviceFeatures;
			vkGetPhysicalDeviceFeatures(device, &deviceFeatures);

			int score = 0;

			// Discrete GPUs have a significant performance advantage
			if (deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
			{
				score += 1000;
			}

			// Maximum possible size of textures affects graphics quality
			score += deviceProperties.limits.maxImageDimension2D;

			bool extensionsSupported = checkDeviceExtensionSupport(device);

			bool swapChainAdequate = false;
			if (extensionsSupported)
			{
				swapChainAdequate = true;
				/*SwapChainSupportDetails swapChainSupport =
				 * querySwapChainSupport(device);*/
				/*swapChainAdequate = !swapChainSupport.formats.empty() &&
				 * !swapChainSupport.presentModes.empty();*/
			}

			QueueMap map = findQueueFamilies(device);

			bool hasGraphicsQueue = map.find(Queue::Type::GRAPHICS) != queueMap.end();
			bool hasComputeQueue  = map.find(Queue::Type::COMPUTE) != queueMap.end();
			bool hasTransferQueue = map.find(Queue::Type::TRANSFER) != queueMap.end();

			// Application can't function without geometry shaders
			if (!deviceFeatures.geometryShader || !hasGraphicsQueue || !hasComputeQueue || !hasTransferQueue ||
				!swapChainAdequate)
			{
				return 0;
			}

			return score;
		}
		QueueMap findQueueFamilies(VkPhysicalDevice device)
		{
			// Assign index to queue families that could be found
			uint32_t queueFamilyCount = 0;
			vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

			std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
			vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

			QueueMap map;

			int i = 0;
			for (const auto &queueFamily : queueFamilies)
			{
				if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT)
				{
					map[Queue::Type::GRAPHICS] = i;
				}
				if (queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT)
				{
					map[Queue::Type::COMPUTE] = i;
				}
				if (queueFamily.queueFlags & VK_QUEUE_TRANSFER_BIT)
				{
					map[Queue::Type::TRANSFER] = i;
				}

				/*VkBool32 presentSupport = false;*/
				/*vkGetPhysicalDeviceSurfaceSupportKHR(device, i, window.surface,
				 * &presentSupport);*/
				/**/
				/*if (presentSupport)*/
				/*{*/
				/*	instance.queueMap[Queue::Type::PRESENTATION] = i;*/
				/*}*/

				i++;
			}

			return map;
		}

		VkShaderModule createShaderModule(const std::vector<char> &code)
		{
			VkShaderModuleCreateInfo createInfo{};
			createInfo.sType	= VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
			createInfo.codeSize = code.size();
			createInfo.pCode	= reinterpret_cast<const uint32_t *>(code.data());

			VkShaderModule shaderModule;
			if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to create shader module!");
			}
			return shaderModule;
		}

		Pipeline createGraphicsPipeline(std::string vertPath, std::string fragPath,
										std::vector<VkDescriptorSetLayout> layouts, VkRenderPass renderPass)
		{
			Pipeline pipeline;

			auto vertShaderCode = readFile(vertPath);
			auto fragShaderCode = readFile(fragPath);

			VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
			VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

			VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
			vertShaderStageInfo.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
			vertShaderStageInfo.stage  = VK_SHADER_STAGE_VERTEX_BIT;
			vertShaderStageInfo.module = vertShaderModule;
			vertShaderStageInfo.pName  = "main";

			VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
			fragShaderStageInfo.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
			fragShaderStageInfo.stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
			fragShaderStageInfo.module = fragShaderModule;
			fragShaderStageInfo.pName  = "main";

			VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

			VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
			vertexInputInfo.sType							= VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
			vertexInputInfo.vertexBindingDescriptionCount	= 0;
			vertexInputInfo.pVertexBindingDescriptions		= nullptr; // Optional
			vertexInputInfo.vertexAttributeDescriptionCount = 0;
			vertexInputInfo.pVertexAttributeDescriptions	= nullptr; // Optional

			VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
			inputAssembly.sType					 = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
			inputAssembly.topology				 = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
			inputAssembly.primitiveRestartEnable = VK_FALSE;

			std::vector<VkDynamicState> dynamicStates = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};

			VkPipelineDynamicStateCreateInfo dynamicState{};
			dynamicState.sType			   = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
			dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
			dynamicState.pDynamicStates	   = dynamicStates.data();

			VkPipelineViewportStateCreateInfo viewportState{};
			viewportState.sType			= VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
			viewportState.viewportCount = 1;
			viewportState.scissorCount	= 1;

			VkPipelineRasterizationStateCreateInfo rasterizer{};
			rasterizer.sType			= VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
			rasterizer.depthClampEnable = VK_FALSE;

			rasterizer.rasterizerDiscardEnable = VK_FALSE;

			rasterizer.polygonMode = VK_POLYGON_MODE_FILL;

			rasterizer.lineWidth = 1.0f;

			rasterizer.cullMode	 = VK_CULL_MODE_NONE;
			rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;

			rasterizer.depthBiasEnable		   = VK_FALSE;
			rasterizer.depthBiasConstantFactor = 0.0f; // Optional
			rasterizer.depthBiasClamp		   = 0.0f; // Optional
			rasterizer.depthBiasSlopeFactor	   = 0.0f; // Optional

			VkPipelineMultisampleStateCreateInfo multisampling{};
			multisampling.sType					= VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
			multisampling.sampleShadingEnable	= VK_FALSE;
			multisampling.rasterizationSamples	= VK_SAMPLE_COUNT_1_BIT;
			multisampling.minSampleShading		= 1.0f;		// Optional
			multisampling.pSampleMask			= nullptr;	// Optional
			multisampling.alphaToCoverageEnable = VK_FALSE; // Optional
			multisampling.alphaToOneEnable		= VK_FALSE; // Optional

			VkPipelineColorBlendAttachmentState colorBlendAttachment{};
			colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
												  VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
			colorBlendAttachment.blendEnable		 = VK_FALSE;
			colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;			// Optional
			colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA; // Optional
			colorBlendAttachment.colorBlendOp		 = VK_BLEND_OP_ADD;						// Optional
			colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;					// Optional
			colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;				// Optional
			colorBlendAttachment.alphaBlendOp		 = VK_BLEND_OP_ADD;						// Optional

			VkPipelineColorBlendStateCreateInfo colorBlending{};
			colorBlending.sType				= VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
			colorBlending.logicOpEnable		= VK_FALSE;
			colorBlending.logicOp			= VK_LOGIC_OP_COPY; // Optional
			colorBlending.attachmentCount	= 1;
			colorBlending.pAttachments		= &colorBlendAttachment;
			colorBlending.blendConstants[0] = 0.0f; // Optional
			colorBlending.blendConstants[1] = 0.0f; // Optional
			colorBlending.blendConstants[2] = 0.0f; // Optional
			colorBlending.blendConstants[3] = 0.0f; // Optional

			VkPushConstantRange pushConstant;
			pushConstant.offset		= 0;
			pushConstant.size		= 0;
			pushConstant.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

			VkPipelineDepthStencilStateCreateInfo depthStencil{};
			depthStencil.sType				   = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
			depthStencil.depthTestEnable	   = VK_TRUE;
			depthStencil.depthWriteEnable	   = VK_TRUE;
			depthStencil.depthCompareOp		   = VK_COMPARE_OP_LESS;
			depthStencil.depthBoundsTestEnable = VK_FALSE;
			depthStencil.minDepthBounds		   = 0.0f; // Optional
			depthStencil.maxDepthBounds		   = 1.0f; // Optional
			depthStencil.stencilTestEnable	   = VK_FALSE;
			depthStencil.front				   = {}; // Optional
			depthStencil.back				   = {}; // Optional

			VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
			pipelineLayoutInfo.sType				  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
			pipelineLayoutInfo.setLayoutCount		  = layouts.size();
			pipelineLayoutInfo.pSetLayouts			  = layouts.data();
			pipelineLayoutInfo.pushConstantRangeCount = 0;		 // Optional
			pipelineLayoutInfo.pPushConstantRanges	  = nullptr; // Optional

			if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipeline.layout) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to create pipeline layout!");
			}

			VkGraphicsPipelineCreateInfo pipelineInfo{};
			pipelineInfo.sType		= VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
			pipelineInfo.stageCount = 2;
			pipelineInfo.pStages	= shaderStages;

			pipelineInfo.pVertexInputState	 = &vertexInputInfo;
			pipelineInfo.pInputAssemblyState = &inputAssembly;
			pipelineInfo.pViewportState		 = &viewportState;
			pipelineInfo.pRasterizationState = &rasterizer;
			pipelineInfo.pMultisampleState	 = &multisampling;
			pipelineInfo.pDepthStencilState	 = &depthStencil; // Optional
			pipelineInfo.pColorBlendState	 = &colorBlending;
			pipelineInfo.pDynamicState		 = &dynamicState;

			pipelineInfo.layout = pipeline.layout;

			pipelineInfo.renderPass = renderPass;
			pipelineInfo.subpass	= 0;

			pipelineInfo.basePipelineHandle = VK_NULL_HANDLE; // Optional
			pipelineInfo.basePipelineIndex	= -1;			  // Optional

			if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline.pipeline) !=
				VK_SUCCESS)
			{
				throw std::runtime_error("failed to create graphics pipeline!");
			}

			vkDestroyShaderModule(device, fragShaderModule, nullptr);
			vkDestroyShaderModule(device, vertShaderModule, nullptr);

			return pipeline;
		}

		DescriptorLayout createLayout(int binding, DescriptorLayout::Type type, DescriptorLayout::Stage stageFlags)
		{
			DescriptorLayout layout;
			layout.type	   = type;
			layout.binding = binding;

			VkDescriptorSetLayoutBinding layoutBinding{};
			layoutBinding.binding			 = binding;
			layoutBinding.descriptorCount	 = 1;
			layoutBinding.descriptorType	 = (VkDescriptorType)type;
			layoutBinding.stageFlags		 = stageFlags;
			layoutBinding.pImmutableSamplers = nullptr;

			VkDescriptorSetLayoutCreateInfo layoutInfo{};
			layoutInfo.sType		= VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
			layoutInfo.bindingCount = 1;
			layoutInfo.pBindings	= &layoutBinding;
			if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &layout.layout) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to create descriptor set layout!");
			}

			return layout;
		}

		DescriptorPool createDescriptorPool()
		{
			return createDescriptorPool(64);
		}

		DescriptorPool createDescriptorPool(ui32 max)
		{
			return createDescriptorPool(max, max, max, max, max, max, max, max, max, max, max, max);
		}

		DescriptorPool createDescriptorPool(ui32 max, ui32 samplerMax, ui32 combinedImageSamplerMax,
											ui32 sampledImageMax, ui32 storageImageMax, ui32 uniformTexelBufferMax,
											ui32 storageTexelBufferMax, ui32 uniformBufferMax, ui32 storageBufferMax,
											ui32 uniformBufferDynamicMax, ui32 storageBufferDynamicMax,
											ui32 inputAttachmentMax)
		{
			DescriptorPool pool;
			pool.device = device;

			VkDescriptorPoolSize poolSize[] = {{VK_DESCRIPTOR_TYPE_SAMPLER, samplerMax},
											   {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, combinedImageSamplerMax},
											   {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, sampledImageMax},
											   {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, storageImageMax},
											   {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, uniformTexelBufferMax},
											   {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, storageTexelBufferMax},
											   {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, uniformBufferMax},
											   {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, storageBufferMax},
											   {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, uniformBufferDynamicMax},
											   {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, storageBufferDynamicMax},
											   {VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, inputAttachmentMax}};

			VkDescriptorPoolCreateInfo poolInfo{};
			poolInfo.sType		   = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
			poolInfo.poolSizeCount = 11;
			poolInfo.pPoolSizes	   = poolSize;

			poolInfo.maxSets = max;

			if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &pool.descriptorPool) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to create descriptor pool!");
			}

			return pool;
		}

		enum BufferUsage
		{
			VERTEX		   = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
			STORAGE_BUFFER = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
			TRANSFER_DST   = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			TRANSFER_SRC   = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		};

		enum BufferAccess
		{
			GPUONLY,
			CPUGPU,
		};

		Buffer createBuffer(int size, unsigned int BufferUsageFlags, BufferAccess access)
		{
			Buffer buffer = vmaAllocator.allocate(size, BufferUsageFlags,
												  {.flags = access == CPUGPU
																? VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
																: (VmaAllocationCreateFlags)0,
												   .usage = VMA_MEMORY_USAGE_AUTO});

			return buffer;
		}

		template <typename T> Buffer createBufferStaged(std::vector<T> &data, ui32 bufferUsageFlags)
		{
			Buffer buffer =
				createBuffer(data.size() * sizeof(T), bufferUsageFlags | Instance::BufferUsage::TRANSFER_DST,
							 Instance::BufferAccess::GPUONLY);

			buffer.elementCount = data.size();

			Buffer stage = createBuffer(data.size() * sizeof(T), Instance::TRANSFER_SRC, CPUGPU);

			stage.singleCopy(data.data());

			copyBufferData(stage, buffer);

			stage.destroy();

			return buffer;
		}

		template <typename T> Buffer createBufferWrite(std::vector<T> &data, ui32 bufferUsageFlags)
		{
			Buffer buffer =
				createBuffer(data.size() * sizeof(T), bufferUsageFlags | Instance::BufferUsage::TRANSFER_DST,
							 Instance::BufferAccess::CPUGPU);

			buffer.elementCount = data.size();

			buffer.singleCopy(data.data());

			return buffer;
		}

		void copyBufferData(Buffer src, Buffer dst)
		{
			// copy
			Command command = transferPool.create();

			command.begin();

			VkBufferCopy copyRegion{};
			copyRegion.srcOffset = 0; // Optional
			copyRegion.dstOffset = 0; // Optional
			copyRegion.size		 = src.size;
			vkCmdCopyBuffer(command.buffer, src.buffer, dst.buffer, 1, &copyRegion);

			command.end(nullptr);
			transferPool.destroy(command);
		}

		Pipeline createComputePipeline(std::string path, std::vector<VkDescriptorSetLayout> layouts)
		{
			VkPipelineShaderStageCreateInfo computeShaderStageInfo{};
			auto							computeShader = readFile("compute.spv");

			VkShaderModuleCreateInfo createInfo{};
			createInfo.sType	= VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
			createInfo.codeSize = computeShader.size();
			createInfo.pCode	= reinterpret_cast<const uint32_t *>(computeShader.data());

			VkShaderModule shader;
			if (vkCreateShaderModule(device, &createInfo, nullptr, &shader) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to create shader module!");
			}

			computeShaderStageInfo.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
			computeShaderStageInfo.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
			computeShaderStageInfo.module = shader;
			computeShaderStageInfo.pName  = "main";

			Pipeline pipeline;

			VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
			pipelineLayoutInfo.sType		  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
			pipelineLayoutInfo.setLayoutCount = layouts.size();
			pipelineLayoutInfo.pSetLayouts	  = layouts.data();

			if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipeline.layout) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to create compute pipeline layout!");
			}

			VkComputePipelineCreateInfo pipelineInfo{};
			pipelineInfo.sType	= VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
			pipelineInfo.layout = pipeline.layout;
			pipelineInfo.stage	= computeShaderStageInfo;

			if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline.pipeline) !=
				VK_SUCCESS)
			{
				throw std::runtime_error("failed to create compute pipeline!");
			}

			return pipeline;
		}

		Queue createQueue(Queue::Type type)
		{
			Queue q;
			q.index	 = queueMap[type];
			q.device = device;

			vkGetDeviceQueue(device, q.index, 0, &q.queue);

			return q;
		}

		Sampler createSampler(VkFilter filter)
		{
			Sampler sampler;

			VkSamplerCreateInfo samplerInfo{};
			samplerInfo.sType	  = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
			samplerInfo.magFilter = filter;
			samplerInfo.minFilter = filter;

			samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
			samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
			samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;

			samplerInfo.anisotropyEnable = VK_TRUE;

			VkPhysicalDeviceProperties properties{};
			vkGetPhysicalDeviceProperties(physicalDevice, &properties);
			samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;

			samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;

			samplerInfo.unnormalizedCoordinates = VK_FALSE;

			samplerInfo.compareEnable = VK_FALSE;
			samplerInfo.compareOp	  = VK_COMPARE_OP_ALWAYS;

			samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
			samplerInfo.mipLodBias = 0.0f;
			samplerInfo.minLod	   = 0.0f;
			samplerInfo.maxLod	   = 0.0f;

			if (vkCreateSampler(device, &samplerInfo, nullptr, &sampler.sampler) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to create texture sampler!");
			}

			return sampler;
		}

		Image createImage(ui32 width, ui32 height, void *data)
		{
			Image image;

			// make staging buffer
			Buffer stagingBuffer = vmaAllocator.allocate(width * height * 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);

			void *map;
			stagingBuffer.map(&map);
			memcpy(map, data, static_cast<size_t>(width * height * 4));
			stagingBuffer.unmap();

			// allocate space for image
			image = vmaAllocator.createImage(width, height, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL,
											 VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
											 {.usage = VMA_MEMORY_USAGE_GPU_ONLY});

			// copy image from buffer
			image.copyFromBuffer(stagingBuffer, transferPool, device, transferQueue.queue);

			// cleanup buffer
			stagingBuffer.destroy();

			return image;
		}
};

class DescriptorManager
{
	public:
		std::vector<DescriptorPool> pools;
		int							poolCursor;
		int							descriptorCursor;
		Instance				   *instance;
		int							maxPoolSize;

		void createNewPool()
		{
			pools.emplace_back();
			poolCursor++;
			descriptorCursor = 0;

			pools[poolCursor] = instance->createDescriptorPool();
		}

		void setup(Instance *instance, int maxPoolSize)
		{
			poolCursor		 = -1;
			descriptorCursor = 0;
			this->instance	 = instance;

			createNewPool();
		}

		Descriptor allocate(DescriptorLayout &layout, Buffer &buffer)
		{
			descriptorCursor++;

			if (descriptorCursor > maxPoolSize)
			{
				createNewPool();

				return allocate(layout, buffer);
			}

			return pools[poolCursor].createDescriptor(layout, &buffer, nullptr, nullptr);
		}

		void clean()
		{
			for (auto &p : pools)
			{
			}
		}
};
