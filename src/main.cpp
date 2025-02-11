#include <algorithm>
#include <array>
#include <chrono>
#include <glm/ext/matrix_transform.hpp>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <thread>
#include <vector>
#include <vulkan/vulkan.h>

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

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

VkCommandBuffer beginSingleTimeCommands(VkCommandPool commandPool, VkDevice device)
{
	VkCommandBufferAllocateInfo allocInfo{};
	allocInfo.sType				 = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	allocInfo.level				 = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	allocInfo.commandPool		 = commandPool;
	allocInfo.commandBufferCount = 1;

	VkCommandBuffer commandBuffer;
	vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

	VkCommandBufferBeginInfo beginInfo{};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

	vkBeginCommandBuffer(commandBuffer, &beginInfo);

	return commandBuffer;
}

void endSingleTimeCommands(VkCommandBuffer commandBuffer, VkCommandPool commandPool, VkDevice device,
						   VkQueue graphicsQueue)
{
	vkEndCommandBuffer(commandBuffer);

	VkSubmitInfo submitInfo{};
	submitInfo.sType			  = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers	  = &commandBuffer;

	vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
	vkQueueWaitIdle(graphicsQueue);

	vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
}

class Command
{
	public:
		VkCommandBuffer buffer;
		VkQueue			queue;
		VkDevice		device;
		VkCommandPool	pool;

		void beginOneTime()
		{
			VkCommandBufferBeginInfo beginInfo{};
			beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

			vkBeginCommandBuffer(buffer, &beginInfo);
		}

		void submit(VkSemaphore semaphores = nullptr, VkPipelineStageFlags flags = 0)
		{
			vkEndCommandBuffer(buffer);

			VkSubmitInfo submitInfo{};
			submitInfo.sType			  = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			submitInfo.commandBufferCount = 1;
			submitInfo.pCommandBuffers	  = &buffer;

			vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
			vkQueueWaitIdle(queue);

			vkFreeCommandBuffers(device, pool, 1, &buffer);
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
			vkEndCommandBuffer(command.buffer);

			VkSubmitInfo submitInfo{};
			submitInfo.sType			  = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			submitInfo.commandBufferCount = 1;
			submitInfo.pCommandBuffers	  = &command.buffer;

			vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
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

class Image
{
	public:
		VkImage		image;
		VkSampler	sampler;
		VkImageView view;

		int width;
		int height;

		VmaAllocation allocation;
		VmaAllocator *allocator;

		void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout,
								   VkCommandPool commandPool, VkDevice device, VkQueue graphicsQueue)
		{
			VkCommandBuffer		 commandBuffer = beginSingleTimeCommands(commandPool, device);
			VkImageMemoryBarrier barrier{};
			barrier.sType	  = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			barrier.oldLayout = oldLayout;
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

			if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
			{
				barrier.srcAccessMask = 0;
				barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

				sourceStage		 = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
				destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
			}
			else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
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

			vkCmdPipelineBarrier(commandBuffer, sourceStage, destinationStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);

			endSingleTimeCommands(commandBuffer, commandPool, device, graphicsQueue);
		}

		void copyFromBuffer(Buffer buffer, VkCommandPool commandPool, VkDevice device, VkQueue graphicsQueue)
		{
			transitionImageLayout(image, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED,
								  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, commandPool, device, graphicsQueue);

			VkCommandBuffer commandBuffer = beginSingleTimeCommands(commandPool, device);

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

			vkCmdCopyBufferToImage(commandBuffer, buffer.buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
								   &region);

			endSingleTimeCommands(commandBuffer, commandPool, device, graphicsQueue);

			transitionImageLayout(image, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
								  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, commandPool, device, graphicsQueue);
		}

		void destroy(VkDevice device)
		{
			vkDestroySampler(device, sampler, nullptr);
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

		Buffer stageAllocate(int size, void *data, unsigned int BUFFER_USAGE, VkCommandPool commandPool,
							 VkQueue graphicsQueue)
		{
			Buffer stagingBuffer = allocate(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);

			void *mappedData;
			stagingBuffer.map(&mappedData);
			memcpy(mappedData, data, size);
			stagingBuffer.unmap();

			Buffer buffer =
				allocate(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT | BUFFER_USAGE, {.usage = VMA_MEMORY_USAGE_GPU_ONLY});

			// copy
			VkCommandBufferAllocateInfo allocInfo{};
			allocInfo.sType				 = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
			allocInfo.level				 = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
			allocInfo.commandPool		 = commandPool;
			allocInfo.commandBufferCount = 1;

			VkCommandBuffer commandBuffer;
			vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

			VkCommandBufferBeginInfo beginInfo{};
			beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

			vkBeginCommandBuffer(commandBuffer, &beginInfo);

			VkBufferCopy copyRegion{};
			copyRegion.srcOffset = 0; // Optional
			copyRegion.dstOffset = 0; // Optional
			copyRegion.size		 = size;
			vkCmdCopyBuffer(commandBuffer, stagingBuffer.buffer, buffer.buffer, 1, &copyRegion);

			vkEndCommandBuffer(commandBuffer);

			VkSubmitInfo submitInfo{};
			submitInfo.sType			  = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			submitInfo.commandBufferCount = 1;
			submitInfo.pCommandBuffers	  = &commandBuffer;

			vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
			vkQueueWaitIdle(graphicsQueue);

			vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);

			stagingBuffer.destroy();

			return buffer;
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

		Image createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling,
						  VkImageUsageFlags		  usage,
						  VmaAllocationCreateInfo allocatorInfo = {
							  .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
							  .usage = VMA_MEMORY_USAGE_CPU_ONLY})
		{
			Image image;
			image.allocator = &vmaAllocator;
			image.width		= width;
			image.height	= height;

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
			viewInfo.subresourceRange.aspectMask	 = VK_IMAGE_ASPECT_COLOR_BIT;
			viewInfo.subresourceRange.baseMipLevel	 = 0;
			viewInfo.subresourceRange.levelCount	 = 1;
			viewInfo.subresourceRange.baseArrayLayer = 0;
			viewInfo.subresourceRange.layerCount	 = 1;

			if (vkCreateImageView(device, &viewInfo, nullptr, &image.view) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to create texture image view!");
			}

			VkSamplerCreateInfo samplerInfo{};
			samplerInfo.sType	  = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
			samplerInfo.magFilter = VK_FILTER_LINEAR;
			samplerInfo.minFilter = VK_FILTER_LINEAR;

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

			if (vkCreateSampler(device, &samplerInfo, nullptr, &image.sampler) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to create texture sampler!");
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

class Window
{
	public:
		GLFWwindow				  *window;
		bool					   framebufferResized = false;
		Queue					   presentQueue;
		VkSurfaceKHR			   surface;
		VkSwapchainKHR			   swapChain;
		std::vector<VkImage>	   swapChainImages;
		VkFormat				   swapChainImageFormat;
		VkExtent2D				   swapChainExtent;
		std::vector<VkImageView>   swapChainImageViews;
		std::vector<VkFramebuffer> swapChainFramebuffers;
		VkRenderPass			   renderPass;

		void setup(int width, int height, std::string name)
		{
			glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
			glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

			window = glfwCreateWindow(width, height, name.c_str(), nullptr, nullptr);

			glfwSetWindowUserPointer(window, this);
			glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
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

struct Vertex
{
		glm::vec4 pos;
		glm::vec4 color;
		glm::vec4 vel;

		static VkVertexInputBindingDescription getBindingDescription()
		{
			VkVertexInputBindingDescription bindingDescription{};
			bindingDescription.binding	 = 0;
			bindingDescription.stride	 = sizeof(Vertex);
			bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

			return bindingDescription;
		}

		static std::array<VkVertexInputAttributeDescription, 3> getAttributeDescriptions()
		{
			std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions{};

			attributeDescriptions[0].binding  = 0;
			attributeDescriptions[0].location = 0;
			attributeDescriptions[0].format	  = VK_FORMAT_R32G32B32_SFLOAT;
			attributeDescriptions[0].offset	  = offsetof(Vertex, pos);

			attributeDescriptions[1].binding  = 0;
			attributeDescriptions[1].location = 1;
			attributeDescriptions[1].format	  = VK_FORMAT_R32G32B32_SFLOAT;
			attributeDescriptions[1].offset	  = offsetof(Vertex, color);

			attributeDescriptions[2].binding  = 0;
			attributeDescriptions[2].location = 2;
			attributeDescriptions[2].format	  = VK_FORMAT_R32G32B32_SFLOAT;
			attributeDescriptions[2].offset	  = offsetof(Vertex, vel);

			return attributeDescriptions;
		}
};

class VertexBuffer
{
	public:
		Buffer buffer;
		int	   vertexCount;

		void destroy(Allocator allocator)
		{
			allocator.free(buffer);
		}
};

struct UniformBufferObject
{
		glm::mat4 model;
		glm::mat4 view;
		glm::mat4 proj;
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

		void setup(VkDescriptorPool descriptorPool, VkDescriptorSetLayout descriptorSetLayout, VkDevice device,
				   Buffer buffer, int size, int binding)
		{
			VkWriteDescriptorSet descriptorWrite = setupMeta(descriptorPool, descriptorSetLayout, device, binding);

			VkDescriptorBufferInfo bufferInfo;
			bufferInfo.buffer = buffer.buffer;
			bufferInfo.range  = size;
			bufferInfo.offset = 0;

			descriptorWrite.descriptorType	 = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			descriptorWrite.pBufferInfo		 = &bufferInfo;
			descriptorWrite.pImageInfo		 = nullptr;
			descriptorWrite.pTexelBufferView = nullptr;

			vkUpdateDescriptorSets(device, 1, &descriptorWrite, 0, nullptr);
		}

		void setup(VkDescriptorPool descriptorPool, VkDescriptorSetLayout descriptorSetLayout, VkDevice device,
				   Image image, int binding)
		{
			VkWriteDescriptorSet descriptorWrite = setupMeta(descriptorPool, descriptorSetLayout, device, binding);

			VkDescriptorImageInfo imageInfo;
			imageInfo.sampler	  = image.sampler;
			imageInfo.imageView	  = image.view;
			imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

			descriptorWrite.descriptorType	 = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			descriptorWrite.pBufferInfo		 = nullptr;
			descriptorWrite.pImageInfo		 = &imageInfo;
			descriptorWrite.pTexelBufferView = nullptr;

			vkUpdateDescriptorSets(device, 1, &descriptorWrite, 0, nullptr);
		}
};

class DescriptorPool
{
	public:
		VkDescriptorPool descriptorPool;
		VkDevice		 device;

		Descriptor createDescriptor(DescriptorLayout descriptorSetLayout, Buffer *buffer, Image *image)
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
			if (image != nullptr)
			{
				imageInfo.sampler	  = image->sampler;
				imageInfo.imageView	  = image->view;
				imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			}

			descriptorWrite.descriptorType	 = (VkDescriptorType)descriptorSetLayout.type;
			descriptorWrite.pBufferInfo		 = buffer != nullptr ? &bufferInfo : nullptr;
			descriptorWrite.pImageInfo		 = image != nullptr ? &imageInfo : nullptr;
			descriptorWrite.pTexelBufferView = nullptr;

			vkUpdateDescriptorSets(device, 1, &descriptorWrite, 0, nullptr);

			return descriptor;
		}
};

class Texture
{
	public:
		Image	   image;
		Descriptor descriptor;
};

class UBO
{
	public:
		Buffer	   buffer;
		Descriptor descriptor;
		void	  *map;

		template <typename T> void update(T data)
		{
			memcpy(map, &data, sizeof(T));
		}
};

class DescriptorBuffer
{
	public:
		VkDescriptorSet descriptorSets;
		Buffer			uniformBuffer;
		void		   *uniformBuffersMapped;

		VkDescriptorSet imageDescriptor;

		VkInstance		 &instance;
		VkDevice		 &device;
		VkPhysicalDevice &physicalDevice;
		Allocator		 &allocator;

		DescriptorBuffer(VkInstance &instance, VkDevice &device, VkPhysicalDevice &physicalDevice, Allocator &allocator)
			: instance(instance), device(device), physicalDevice(physicalDevice), allocator(allocator)
		{
		}

		void create(VkDescriptorPool descriptorPool, VkDescriptorSetLayout descriptorSetLayout, Image image,
					VkDescriptorSetLayout textureLayout)
		{
			// UNIFORM
			VkDeviceSize bufferSize = sizeof(UniformBufferObject);

			uniformBuffer = allocator.allocate(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

			uniformBuffer.map(&uniformBuffersMapped);

			// SET

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
			}
			{
				VkDescriptorSetLayout		layouts = textureLayout;
				VkDescriptorSetAllocateInfo allocInfo{};
				allocInfo.sType				 = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
				allocInfo.descriptorPool	 = descriptorPool;
				allocInfo.descriptorSetCount = 1;
				allocInfo.pSetLayouts		 = &layouts;

				if (vkAllocateDescriptorSets(device, &allocInfo, &imageDescriptor) != VK_SUCCESS)
				{
					throw std::runtime_error("failed to allocate descriptor sets!");
				}
			}

			{
				VkDescriptorBufferInfo bufferInfo{};
				bufferInfo.buffer = uniformBuffer.buffer;
				bufferInfo.offset = 0;
				bufferInfo.range  = sizeof(UniformBufferObject);

				VkWriteDescriptorSet descriptorWrite;
				descriptorWrite.sType			 = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				descriptorWrite.dstSet			 = descriptorSets;
				descriptorWrite.dstBinding		 = 0;
				descriptorWrite.dstArrayElement	 = 0;
				descriptorWrite.descriptorType	 = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
				descriptorWrite.descriptorCount	 = 1;
				descriptorWrite.pBufferInfo		 = &bufferInfo;
				descriptorWrite.pImageInfo		 = nullptr; // Optional
				descriptorWrite.pTexelBufferView = nullptr; // optional

				vkUpdateDescriptorSets(device, 1, &descriptorWrite, 0, nullptr);
			}
			{
				VkDescriptorImageInfo imageInfo;
				imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
				imageInfo.imageView	  = image.view;
				imageInfo.sampler	  = image.sampler;

				VkWriteDescriptorSet descriptorWrite;
				descriptorWrite.sType			= VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				descriptorWrite.dstSet			= imageDescriptor;
				descriptorWrite.dstBinding		= 1;
				descriptorWrite.dstArrayElement = 0;
				descriptorWrite.descriptorType	= VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
				descriptorWrite.descriptorCount = 1;
				descriptorWrite.pImageInfo		= &imageInfo;

				vkUpdateDescriptorSets(device, 1, &descriptorWrite, 0, nullptr);
			}
		}

		void destroy()
		{
			uniformBuffer.unmap();
			uniformBuffer.destroy();
		}

		void update(UniformBufferObject ubo)
		{
			memcpy(uniformBuffersMapped, &ubo, sizeof(ubo));
		}
};

class Pipeline
{
	public:
		VkPipeline		 pipeline;
		VkPipelineLayout layout;
		CommandPool		 pool;
		Command			 command;
		Queue			 queue;

		void startDispatch()
		{
			VkCommandBufferBeginInfo beginInfo{};
			beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

			if (vkBeginCommandBuffer(command.buffer, &beginInfo) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to begin recording command buffer!");
			}

			vkCmdBindPipeline(command.buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
		}
		void dispatch(std::vector<VkDescriptorSet> descriptor, int x, int y, int z)
		{
			vkCmdBindDescriptorSets(command.buffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, descriptor.size(),
									descriptor.data(), 0, 0);

			vkCmdDispatch(command.buffer, x, y, z);
		}
		void endDispatch()
		{
			if (vkEndCommandBuffer(command.buffer) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to record command buffer!");
			}

			VkSubmitInfo submitInfo{};
			submitInfo.sType			  = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			submitInfo.commandBufferCount = 1;
			submitInfo.pCommandBuffers	  = &command.buffer;
			if (vkQueueSubmit(queue.queue, 1, &submitInfo, nullptr) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to submit compute command buffer!");
			}
			vkQueueWaitIdle(queue.queue);
		}
};

class GraphicsPipeline
{
	public:
		VkPipeline		 pipeline;
		VkPipelineLayout layout;
		Queue			 graphicsQueue;
		CommandPool		 graphicsPool;
		Command			 graphicsBuffer;
};

typedef std::map<Queue::Type, uint32_t> QueueMap;

class Instance
{
	public:
		VkInstance				 instance;
		VkDebugUtilsMessengerEXT debugMessenger;
		VkPhysicalDevice		 physicalDevice = VK_NULL_HANDLE;
		VkDevice				 device;
		QueueMap				 queueMap;
		std::vector<uint32_t>	 uniqueQueues;
		CommandPool				 graphicsPool;
		Queue					 transferQueue;
		CommandPool				 transferPool;
		VkDescriptorPool		 descriptorPool;
		Allocator				 vmaAllocator;

#ifdef NDEBUG
		const bool enableValidationLayers = false;
#else
		const bool enableValidationLayers = true;
#endif

		const std::vector<const char *> validationLayers = {"VK_LAYER_KHRONOS_validation"};
		const std::vector<const char *> deviceExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

		Instance()
		{
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

			std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

			return VK_FALSE;
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

		void bootstrap()
		{
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

			LOG(hasGraphicsQueue);
			LOG(hasComputeQueue);
			LOG(hasTransferQueue);

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

		GraphicsPipeline createGraphicsPipeline(std::string vertPath, std::string fragPath,
												std::vector<VkDescriptorSetLayout> layouts, VkRenderPass renderPass)
		{
			GraphicsPipeline pipeline;

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

			auto bindDesc = Vertex::getBindingDescription();
			auto attrDesc = Vertex::getAttributeDescriptions();

			VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
			vertexInputInfo.sType							= VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
			vertexInputInfo.vertexBindingDescriptionCount	= 1;
			vertexInputInfo.pVertexBindingDescriptions		= &bindDesc; // Optional
			vertexInputInfo.vertexAttributeDescriptionCount = attrDesc.size();
			vertexInputInfo.pVertexAttributeDescriptions	= attrDesc.data(); // Optional

			VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
			inputAssembly.sType					 = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
			inputAssembly.topology				 = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
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
			rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

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
			pushConstant.size		= sizeof(UniformBufferObject);
			pushConstant.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

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
			pipelineInfo.pDepthStencilState	 = nullptr; // Optional
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

			pipeline.graphicsQueue	= this->createQueue(Queue::Type::GRAPHICS);
			pipeline.graphicsPool	= pipeline.graphicsQueue.createCommandPool();
			pipeline.graphicsBuffer = pipeline.graphicsPool.create();

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
			DescriptorPool pool;
			pool.device = device;

			VkDescriptorPoolSize poolSize[] = {
				{.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 16},
				{.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 16},
				{.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 16},
			};

			VkDescriptorPoolCreateInfo poolInfo{};
			poolInfo.sType		   = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
			poolInfo.poolSizeCount = 3;
			poolInfo.pPoolSizes	   = poolSize;

			poolInfo.maxSets = 64;

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

		void copyBufferData(Buffer src, Buffer dst)
		{
			// copy
			Command command = transferPool.create();

			command.beginOneTime();

			VkBufferCopy copyRegion{};
			copyRegion.srcOffset = 0; // Optional
			copyRegion.dstOffset = 0; // Optional
			copyRegion.size		 = src.size;
			vkCmdCopyBuffer(command.buffer, src.buffer, dst.buffer, 1, &copyRegion);

			command.submit();
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

			pipeline.queue	 = this->createQueue(Queue::Type::COMPUTE);
			pipeline.pool	 = pipeline.queue.createCommandPool();
			pipeline.command = pipeline.pool.create();

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

		VertexBuffer createVertexBuffer(std::vector<Vertex> data)
		{
			VertexBuffer vertexBuffer;

			vertexBuffer.vertexCount = data.size();

			vertexBuffer.buffer = vmaAllocator.stageAllocate(sizeof(data[0]) * data.size(), data.data(),
															 VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
															 transferPool.commandPool, transferQueue.queue);

			return vertexBuffer;
		}

		template <typename T> UBO createUBO(VkDescriptorPool descriptorPool, VkDescriptorSetLayout descriptorSetLayout)
		{
			UBO ubo;
			ubo.buffer = vmaAllocator.allocate(sizeof(T), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
			ubo.buffer.map(&ubo.map);

			ubo.descriptor.setup(descriptorPool, descriptorSetLayout, device, ubo.buffer, sizeof(T), 0);

			return ubo;
		}

		Texture createTexture(int width, int height, void *data, VkDescriptorPool pool, VkDescriptorSetLayout layout)
		{
			Texture texture;

			// make staging buffer
			Buffer stagingBuffer = vmaAllocator.allocate(width * height * 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);

			void *map;
			stagingBuffer.map(&map);
			memcpy(map, data, static_cast<size_t>(width * height * 4));
			stagingBuffer.unmap();

			// allocate space for image
			texture.image = vmaAllocator.createImage(width, height, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL,
													 VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
													 {.usage = VMA_MEMORY_USAGE_GPU_ONLY});

			// copy image from buffer
			texture.image.copyFromBuffer(stagingBuffer, transferPool.commandPool, device, transferQueue.queue);

			// cleanup buffer
			stagingBuffer.destroy();

			texture.descriptor.setup(pool, layout, device, texture.image, 0);

			return texture;
		}

		void destroyUBO(UBO &ubo)
		{
			ubo.buffer.unmap();
			ubo.buffer.destroy();
		}
};

class HelloTriangleApplication
{
	public:
		void run()
		{
			initWindow();
			initVulkan();
			mainLoop();
			cleanup();
		}
		HelloTriangleApplication()
		{
		}

	private:
		Window				  window;
		Instance			  instance;
		VkDescriptorSetLayout descriptorSetLayout;
		GraphicsPipeline	  pipeline;

		uint32_t imageIndex;

		VkDescriptorPool descriptorPool;

		VkDescriptorSetLayout textureLayout;
		VkDescriptorSet		  textureSet;

		VkDescriptorSetLayout compLayoutIn;
		VkDescriptorSetLayout compLayoutOut;

		VkSemaphore imageAvailableSemaphores;
		VkSemaphore renderFinishedSemaphores;
		VkFence		inFlightFences;

		uint32_t currentFrame = 0;

#ifdef NDEBUG
		const bool enableValidationLayers = false;
#else
		const bool enableValidationLayers = true;
#endif

		void initWindow()
		{
		}

		void createInstance()
		{
			instance.bootstrap();
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

			std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

			return VK_FALSE;
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

		void setupDebugMessenger()
		{
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

			bool hasGraphicsQueue = map.find(Queue::Type::GRAPHICS) != instance.queueMap.end();
			bool hasComputeQueue  = map.find(Queue::Type::COMPUTE) != instance.queueMap.end();
			bool hasTransferQueue = map.find(Queue::Type::TRANSFER) != instance.queueMap.end();

			LOG(hasGraphicsQueue);
			LOG(hasComputeQueue);
			LOG(hasTransferQueue);

			// Application can't function without geometry shaders
			if (!deviceFeatures.geometryShader || !hasGraphicsQueue || !hasComputeQueue || !hasTransferQueue ||
				!swapChainAdequate)
			{
				return 0;
			}

			return score;
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

		VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR &capabilities)
		{
			if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
			{
				return capabilities.currentExtent;
			}
			else
			{
				int width, height;
				window.getFrameBufferSize(&width, &height);

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

		bool checkDeviceExtensionSupport(VkPhysicalDevice device)
		{
			uint32_t extensionCount;
			vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

			std::vector<VkExtensionProperties> availableExtensions(extensionCount);
			vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

			std::set<std::string> requiredExtensions(instance.deviceExtensions.begin(),
													 instance.deviceExtensions.end());

			for (const auto &extension : availableExtensions)
			{
				requiredExtensions.erase(extension.extensionName);
			}

			return requiredExtensions.empty();
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

		void pickPhysicalDevice()
		{
			uint32_t deviceCount = 0;

			vkEnumeratePhysicalDevices(instance.instance, &deviceCount, nullptr);

			if (deviceCount == 0)
			{
				throw std::runtime_error("failed to find GPUs with Vulkan support!");
			}

			std::vector<VkPhysicalDevice> devices(deviceCount);
			vkEnumeratePhysicalDevices(instance.instance, &deviceCount, devices.data());

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
				instance.physicalDevice = candidates.rbegin()->second;
				instance.queueMap		= findQueueFamilies(instance.physicalDevice);
			}
			else
			{
				throw std::runtime_error("failed to find a suitable GPU!");
			}
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

			vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, window.surface, &details.capabilities);

			uint32_t formatCount;
			vkGetPhysicalDeviceSurfaceFormatsKHR(device, window.surface, &formatCount, nullptr);

			if (formatCount != 0)
			{
				details.formats.resize(formatCount);
				vkGetPhysicalDeviceSurfaceFormatsKHR(device, window.surface, &formatCount, details.formats.data());
			}

			uint32_t presentModeCount;
			vkGetPhysicalDeviceSurfacePresentModesKHR(device, window.surface, &presentModeCount, nullptr);

			if (presentModeCount != 0)
			{
				details.presentModes.resize(presentModeCount);
				vkGetPhysicalDeviceSurfacePresentModesKHR(device, window.surface, &presentModeCount,
														  details.presentModes.data());
			}

			return details;
		}

		void createLogicalDevice()
		{
			std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;

			std::set<uint32_t> unique;
			for (auto &key : instance.queueMap)
			{
				unique.insert(key.second);
			}

			for (auto e : unique)
			{
				instance.uniqueQueues.push_back(e);
			}

			float queuePriority = 1.0f;
			for (auto &uniqueQueueFamily : instance.uniqueQueues)
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

			createInfo.enabledExtensionCount   = static_cast<uint32_t>(instance.deviceExtensions.size());
			createInfo.ppEnabledExtensionNames = instance.deviceExtensions.data();

			if (enableValidationLayers)
			{
				createInfo.enabledLayerCount   = static_cast<uint32_t>(instance.validationLayers.size());
				createInfo.ppEnabledLayerNames = instance.validationLayers.data();
			}
			else
			{
				createInfo.enabledLayerCount = 0;
			}

			if (vkCreateDevice(instance.physicalDevice, &createInfo, nullptr, &instance.device) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to create logical device!");
			}
		}

		void createSwapChain()
		{
			SwapChainSupportDetails swapChainSupport = querySwapChainSupport(instance.physicalDevice);

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
			createInfo.surface = window.surface;

			createInfo.minImageCount	= imageCount;
			createInfo.imageFormat		= surfaceFormat.format;
			createInfo.imageColorSpace	= surfaceFormat.colorSpace;
			createInfo.imageExtent		= extent;
			createInfo.imageArrayLayers = 1;
			createInfo.imageUsage		= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

			if (instance.uniqueQueues.size() > 1)
			{
				createInfo.imageSharingMode		 = VK_SHARING_MODE_CONCURRENT;
				createInfo.queueFamilyIndexCount = instance.uniqueQueues.size();
				createInfo.pQueueFamilyIndices	 = instance.uniqueQueues.data();
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

			if (vkCreateSwapchainKHR(instance.device, &createInfo, nullptr, &window.swapChain) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to create swap chain!");
			}

			vkGetSwapchainImagesKHR(instance.device, window.swapChain, &imageCount, nullptr);
			window.swapChainImages.resize(imageCount);
			vkGetSwapchainImagesKHR(instance.device, window.swapChain, &imageCount, window.swapChainImages.data());

			window.swapChainImageFormat = surfaceFormat.format;
			window.swapChainExtent		= extent;
		}

		void initVulkan()
		{
		}

		VkImageView createImageView(VkImage image, VkFormat format)
		{
			VkImageViewCreateInfo viewInfo{};
			viewInfo.sType							 = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			viewInfo.image							 = image;
			viewInfo.viewType						 = VK_IMAGE_VIEW_TYPE_2D;
			viewInfo.format							 = format;
			viewInfo.subresourceRange.aspectMask	 = VK_IMAGE_ASPECT_COLOR_BIT;
			viewInfo.subresourceRange.baseMipLevel	 = 0;
			viewInfo.subresourceRange.levelCount	 = 1;
			viewInfo.subresourceRange.baseArrayLayer = 0;
			viewInfo.subresourceRange.layerCount	 = 1;

			VkImageView imageView;
			if (vkCreateImageView(instance.device, &viewInfo, nullptr, &imageView) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to create texture image view!");
			}

			return imageView;
		}

		void createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling,
						 VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage &image,
						 VkDeviceMemory &imageMemory)
		{
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

			if (vkCreateImage(instance.device, &imageInfo, nullptr, &image) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to create image!");
			}

			VkMemoryRequirements memRequirements;
			vkGetImageMemoryRequirements(instance.device, image, &memRequirements);

			VkMemoryAllocateInfo allocInfo{};
			allocInfo.sType			  = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
			allocInfo.allocationSize  = memRequirements.size;
			allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

			if (vkAllocateMemory(instance.device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to allocate image memory!");
			}

			vkBindImageMemory(instance.device, image, imageMemory, 0);
		}

		void createDescriptorPool()
		{
			VkDescriptorPoolSize poolSize[] = {
				{.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 2},
				{.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1},
				{.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 2},
			};

			VkDescriptorPoolCreateInfo poolInfo{};
			poolInfo.sType		   = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
			poolInfo.poolSizeCount = 3;
			poolInfo.pPoolSizes	   = poolSize;

			poolInfo.maxSets = 30;

			if (vkCreateDescriptorPool(instance.device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to create descriptor pool!");
			}
		}

		void layoutMaker9000(VkDescriptorSetLayout &layout, int binding, VkDescriptorType type,
							 VkShaderStageFlags flags)
		{
			VkDescriptorSetLayoutBinding layoutBinding{};
			layoutBinding.binding			 = binding;
			layoutBinding.descriptorCount	 = 1;
			layoutBinding.descriptorType	 = type;
			layoutBinding.stageFlags		 = flags;
			layoutBinding.pImmutableSamplers = nullptr;

			VkDescriptorSetLayoutCreateInfo layoutInfo{};
			layoutInfo.sType		= VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
			layoutInfo.bindingCount = 1;
			layoutInfo.pBindings	= &layoutBinding;
			if (vkCreateDescriptorSetLayout(instance.device, &layoutInfo, nullptr, &layout) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to create descriptor set layout!");
			}
		}

		void createDescriptorSetLayout()
		{
			layoutMaker9000(descriptorSetLayout, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT);
			layoutMaker9000(textureLayout, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT);
			/*layoutMaker9000(compLayoutIn, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			 * VK_SHADER_STAGE_COMPUTE_BIT);*/
			/*layoutMaker9000(compLayoutOut, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			 * VK_SHADER_STAGE_COMPUTE_BIT);*/
		}

		void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,
						  VkBuffer &buffer, VkDeviceMemory &bufferMemory)
		{
			VkBufferCreateInfo bufferInfo{};
			bufferInfo.sType	   = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
			bufferInfo.size		   = size;
			bufferInfo.usage	   = usage;
			bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

			if (vkCreateBuffer(instance.device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to create buffer!");
			}

			VkMemoryRequirements memRequirements;
			vkGetBufferMemoryRequirements(instance.device, buffer, &memRequirements);

			VkMemoryAllocateInfo allocInfo{};
			allocInfo.sType			  = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
			allocInfo.allocationSize  = memRequirements.size;
			allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

			if (vkAllocateMemory(instance.device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to allocate buffer memory!");
			}

			vkBindBufferMemory(instance.device, buffer, bufferMemory, 0);
		}

		void createVertexBuffer()
		{
			instance.vmaAllocator.setup(instance.physicalDevice, instance.device, instance.instance);

			/*square.create(graphicsQueue, commandPool, physicalDevice, device,
			 * instance, vmaAllocator, squareData);*/
		}

		uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties)
		{
			VkPhysicalDeviceMemoryProperties memProperties;
			vkGetPhysicalDeviceMemoryProperties(instance.physicalDevice, &memProperties);
			for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
			{
				if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
				{
					return i;
				}
			}

			throw std::runtime_error("failed to find suitable memory type!");
		}

		void cleanupSwapChain()
		{
			for (size_t i = 0; i < window.swapChainFramebuffers.size(); i++)
			{
				vkDestroyFramebuffer(instance.device, window.swapChainFramebuffers[i], nullptr);
			}

			for (size_t i = 0; i < window.swapChainImageViews.size(); i++)
			{
				vkDestroyImageView(instance.device, window.swapChainImageViews[i], nullptr);
			}

			vkDestroySwapchainKHR(instance.device, window.swapChain, nullptr);
		}

		void recreateSwapChain()
		{
			int width = 0, height = 0;
			window.getFrameBufferSize(&width, &height);
			while (width == 0 || height == 0)
			{
				window.getFrameBufferSize(&width, &height);
				window.sleepTillEvent();
			}

			vkDeviceWaitIdle(instance.device);

			cleanupSwapChain();

			createSwapChain();
			createImageViews();
			createFrameBuffers();
		}

		void createSyncObjects()
		{
			VkSemaphoreCreateInfo semaphoreInfo{};
			semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

			VkFenceCreateInfo fenceInfo{};
			fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
			fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

			if (vkCreateSemaphore(instance.device, &semaphoreInfo, nullptr, &imageAvailableSemaphores) != VK_SUCCESS ||
				vkCreateSemaphore(instance.device, &semaphoreInfo, nullptr, &renderFinishedSemaphores) != VK_SUCCESS ||
				vkCreateFence(instance.device, &fenceInfo, nullptr, &inFlightFences) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to create semaphores!");
			}
		}

		void createCommandPool()
		{
		}

		void createFrameBuffers()
		{
			window.swapChainFramebuffers.resize(window.swapChainImageViews.size());

			for (size_t i = 0; i < window.swapChainImageViews.size(); i++)
			{
				VkImageView attachments[] = {window.swapChainImageViews[i]};

				VkFramebufferCreateInfo framebufferInfo{};
				framebufferInfo.sType			= VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
				framebufferInfo.renderPass		= window.renderPass;
				framebufferInfo.attachmentCount = 1;
				framebufferInfo.pAttachments	= attachments;
				framebufferInfo.width			= window.swapChainExtent.width;
				framebufferInfo.height			= window.swapChainExtent.height;
				framebufferInfo.layers			= 1;

				if (vkCreateFramebuffer(instance.device, &framebufferInfo, nullptr, &window.swapChainFramebuffers[i]) !=
					VK_SUCCESS)
				{
					throw std::runtime_error("failed to create framebuffer!");
				}
			}
		}

		void createRenderPass()
		{
			VkAttachmentDescription colorAttachment{};
			colorAttachment.format	= window.swapChainImageFormat;
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

			VkSubpassDescription subpass{};
			subpass.pipelineBindPoint	 = VK_PIPELINE_BIND_POINT_GRAPHICS;
			subpass.colorAttachmentCount = 1;
			subpass.pColorAttachments	 = &colorAttachmentRef;

			VkSubpassDependency dependency{};
			dependency.srcSubpass	 = VK_SUBPASS_EXTERNAL;
			dependency.dstSubpass	 = 0;
			dependency.srcStageMask	 = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
			dependency.srcAccessMask = 0;
			dependency.dstStageMask	 = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
			dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

			VkRenderPassCreateInfo renderPassInfo{};
			renderPassInfo.sType		   = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
			renderPassInfo.attachmentCount = 1;
			renderPassInfo.pAttachments	   = &colorAttachment;
			renderPassInfo.subpassCount	   = 1;
			renderPassInfo.pSubpasses	   = &subpass;
			renderPassInfo.dependencyCount = 1;
			renderPassInfo.pDependencies   = &dependency;

			if (vkCreateRenderPass(instance.device, &renderPassInfo, nullptr, &window.renderPass) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to create render pass!");
			}

			std::cout << window.renderPass << '\n';
		}

		void createImageViews()
		{
			window.swapChainImageViews.resize(window.swapChainImages.size());
			for (size_t i = 0; i < window.swapChainImages.size(); i++)
			{
				VkImageViewCreateInfo createInfo{};
				createInfo.sType						   = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
				createInfo.image						   = window.swapChainImages[i];
				createInfo.viewType						   = VK_IMAGE_VIEW_TYPE_2D;
				createInfo.format						   = window.swapChainImageFormat;
				createInfo.components.r					   = VK_COMPONENT_SWIZZLE_IDENTITY;
				createInfo.components.g					   = VK_COMPONENT_SWIZZLE_IDENTITY;
				createInfo.components.b					   = VK_COMPONENT_SWIZZLE_IDENTITY;
				createInfo.components.a					   = VK_COMPONENT_SWIZZLE_IDENTITY;
				createInfo.subresourceRange.aspectMask	   = VK_IMAGE_ASPECT_COLOR_BIT;
				createInfo.subresourceRange.baseMipLevel   = 0;
				createInfo.subresourceRange.levelCount	   = 1;
				createInfo.subresourceRange.baseArrayLayer = 0;
				createInfo.subresourceRange.layerCount	   = 1;

				if (vkCreateImageView(instance.device, &createInfo, nullptr, &window.swapChainImageViews[i]) !=
					VK_SUCCESS)
				{
					throw std::runtime_error("failed to create image views!");
				}
			}
		}

		VkShaderModule createShaderModule(const std::vector<char> &code)
		{
			VkShaderModuleCreateInfo createInfo{};
			createInfo.sType	= VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
			createInfo.codeSize = code.size();
			createInfo.pCode	= reinterpret_cast<const uint32_t *>(code.data());

			VkShaderModule shaderModule;
			if (vkCreateShaderModule(instance.device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to create shader module!");
			}
			return shaderModule;
		}

		void createGraphicsPipeline()
		{
			pipeline = instance.createGraphicsPipeline("vert.spv", "frag.spv", {descriptorSetLayout, textureLayout},
													   window.renderPass);
		}

		void mainLoop()
		{
			glfwInit();

			instance.bootstrap();

			window.setup(800, 600, "Vizium");
			window.surface = window.createWindowSurface(instance.instance);
			createSwapChain();
			createImageViews();
			createRenderPass();
			createFrameBuffers();
			window.presentQueue = instance.transferQueue;

			createDescriptorSetLayout();
			createGraphicsPipeline();
			createSyncObjects();

			instance.vmaAllocator.setup(instance.physicalDevice, instance.device, instance.instance);
			createDescriptorPool();

			VkPhysicalDeviceProperties prop;
			vkGetPhysicalDeviceProperties(instance.physicalDevice, &prop);

			std::cout << prop.limits.maxComputeWorkGroupInvocations << '\n';

			std::vector<Vertex> triangleData;

			int distpa		= 1024;
			int pointAmount = 1024 * distpa;

			std::default_random_engine	   rand(1);
			std::uniform_real_distribution dist(-1., 1.);
			std::uniform_real_distribution colDist(0., 1.);
			for (int i = 0; i < pointAmount; i++)
			{
				Vertex v;
				v.pos.x = .1 + (float)i / pointAmount / 1.5;
				v.pos.y = dist(rand) / 5;
				v.pos.z = 0;
				v.pos.w = 0;

				v.color.x = colDist(rand);
				v.color.y = colDist(rand);
				v.color.z = colDist(rand);
				v.color.w = 0;

				v.vel.x = 0;
				v.vel.y = -0.001;
				v.vel.z = 0;
				v.vel.w = 0;

				triangleData.push_back(v);
			}

			Buffer shaderStorageBuffer1 =
				instance.createBuffer(triangleData.size() * sizeof(Vertex),
									  Instance::BufferUsage::STORAGE_BUFFER | Instance::BufferUsage::VERTEX |
										  Instance::BufferUsage::TRANSFER_DST,
									  Instance::BufferAccess::GPUONLY);

			{
				Buffer stage = instance.createBuffer(triangleData.size() * sizeof(Vertex), Instance::TRANSFER_SRC,
													 Instance::BufferAccess::CPUGPU);
				stage.singleCopy(triangleData.data());

				instance.copyBufferData(stage, shaderStorageBuffer1);

				stage.destroy();
			}

			Buffer shaderStorageBuffer2 = instance.createBuffer(
				triangleData.size() * sizeof(Vertex),
				Instance::BufferUsage::VERTEX | Instance::BufferUsage::STORAGE_BUFFER, Instance::BufferAccess::CPUGPU);

			DescriptorLayout compLayoutIn =
				instance.createLayout(0, DescriptorLayout::Type::STORAGE_BUFFER, DescriptorLayout::Stage::COMPUTE);
			DescriptorLayout compLayoutOut =
				instance.createLayout(0, DescriptorLayout::Type::STORAGE_BUFFER, DescriptorLayout::Stage::COMPUTE);

			DescriptorPool p;
			p.descriptorPool = descriptorPool;
			p.device		 = instance.device;

			Descriptor compIn  = p.createDescriptor(compLayoutIn, &shaderStorageBuffer1, nullptr);
			Descriptor compOut = p.createDescriptor(compLayoutOut, &shaderStorageBuffer2, nullptr);

			/*Descriptor compIn;*/
			/*{*/
			/*	auto write = compIn.setupMeta(descriptorPool, compLayoutIn.layout,
			 * instance.device, 0);*/
			/**/
			/*	VkDescriptorBufferInfo bufferInfo;*/
			/*	bufferInfo.buffer = shaderStorageBuffer1.buffer;*/
			/*	bufferInfo.range  = sizeof(Vertex) * triangleData.size();*/
			/*	bufferInfo.offset = 0;*/
			/**/
			/*	write.descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;*/
			/*	write.pBufferInfo	   = &bufferInfo;*/
			/*	write.pImageInfo	   = nullptr;*/
			/*	write.pTexelBufferView = nullptr;*/
			/**/
			/*	vkUpdateDescriptorSets(instance.device, 1, &write, 0, nullptr);*/
			/*}*/
			/*Descriptor compOut;*/
			/*{*/
			/*	auto write = compOut.setupMeta(descriptorPool, compLayoutOut.layout,
			 * instance.device, 0);*/
			/**/
			/*	VkDescriptorBufferInfo bufferInfo;*/
			/*	bufferInfo.buffer = shaderStorageBuffer2.buffer;*/
			/*	bufferInfo.range  = sizeof(Vertex) * triangleData.size();*/
			/*	bufferInfo.offset = 0;*/
			/**/
			/*	write.descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;*/
			/*	write.pBufferInfo	   = &bufferInfo;*/
			/*	write.pImageInfo	   = nullptr;*/
			/*	write.pTexelBufferView = nullptr;*/
			/**/
			/*	vkUpdateDescriptorSets(instance.device, 1, &write, 0, nullptr);*/
			/*}*/

			Pipeline compPipeline =
				instance.createComputePipeline("compute.spv", {compLayoutIn.layout, compLayoutOut.layout});

			VertexBuffer triangle = instance.createVertexBuffer(triangleData);

			UBO uniform = instance.createUBO<UniformBufferObject>(descriptorPool, descriptorSetLayout);

			// load image from file
			int			 texWidth, texHeight, texChannels;
			stbi_uc		*pixels	   = stbi_load("./texture.jpg", &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
			VkDeviceSize imageSize = texWidth * texHeight * 4;

			if (!pixels)
			{
				throw std::runtime_error("failed to load texture image!");
			}

			Texture texture = instance.createTexture(texWidth, texHeight, pixels, descriptorPool, textureLayout);

			int frame = 0;

			bool flip = true;

			while (!window.shouldClose())
			{
				std::this_thread::sleep_for(std::chrono::milliseconds(20));
				auto start = std::chrono::system_clock::now();
				glfwPollEvents();

				startDraw();

				static float shift = 0;

				if (glfwGetKey(window.window, GLFW_KEY_SPACE))
				{
					shift += 0.01;
				}

				std::vector desc = flip ? std::vector{compIn.descriptorSets, compOut.descriptorSets}
										: std::vector{compOut.descriptorSets, compIn.descriptorSets};

				compPipeline.startDispatch();
				compPipeline.dispatch(desc, distpa, 1, 1);
				compPipeline.endDispatch();

				UniformBufferObject ubo{};
				ubo.model = glm::translate(glm::mat4(1.f), {0, 0, 0});
				ubo.model = glm::scale(ubo.model, {1, 1, 1});

				ubo.model = glm::rotate(ubo.model, frame * glm::radians(0.0f), glm::vec3(0.0f, 0.0f, 1.0f));

				ubo.view =
					glm::lookAt(glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));

				ubo.proj =
					glm::perspective((float)glm::radians(45.0f),
									 (float)window.swapChainExtent.width / window.swapChainExtent.height, 0.1f, 100.0f);

				ubo.proj = glm::ortho<float>(-1, 1, -1, 1);
				/*ubo.proj = glm::mat4(1);*/
				/*ubo.proj[1][1] *= -1;*/

				/*ubo = UniformBufferObject();*/

				uniform.update(ubo);

				VertexBuffer vb;
				vb.buffer.buffer = flip ? shaderStorageBuffer2.buffer : shaderStorageBuffer1.buffer;
				vb.vertexCount	 = triangleData.size();
				draw(vb, {uniform.descriptor, texture.descriptor});

				ubo.model = glm::translate(glm::mat4(1.f), {0, 0, 0});
				endDraw();

				auto end = std::chrono::system_clock::now();
				frame++;

				flip = !flip;
			}

			Vertex *map;
			shaderStorageBuffer2.map((void **)&map);
			for (int i = 0; i < 3; i++)
			{
				std::cout << map[i].pos.x << " " << map[i].pos.y << " " << map[i].pos.z << " " << map[i].pos.w << "\n";
				std::cout << map[i].color.x << " " << map[i].color.y << " " << map[i].color.z << " " << map[i].color.w
						  << "\n";
				std::cout << "\n";
			}

			vkDeviceWaitIdle(instance.device);

			instance.destroyUBO(uniform);
			/*triangle.destroy(instance.vmaAllocator);*/
		}

		void draw(VertexBuffer vertexBuffer, std::vector<Descriptor> descriptors)
		{
			vkCmdBindPipeline(pipeline.graphicsBuffer.buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.pipeline);

			VkViewport viewport{};
			viewport.x		  = 0.0f;
			viewport.y		  = 0.0f;
			viewport.width	  = static_cast<float>(window.swapChainExtent.width);
			viewport.height	  = static_cast<float>(window.swapChainExtent.height);
			viewport.minDepth = 0.0f;
			viewport.maxDepth = 1.0f;
			vkCmdSetViewport(pipeline.graphicsBuffer.buffer, 0, 1, &viewport);

			VkRect2D scissor{};
			scissor.offset = {0, 0};
			scissor.extent = window.swapChainExtent;
			vkCmdSetScissor(pipeline.graphicsBuffer.buffer, 0, 1, &scissor);

			VkBuffer	 vertexBuffers[] = {vertexBuffer.buffer.buffer};
			VkDeviceSize offsets[]		 = {0};
			vkCmdBindVertexBuffers(pipeline.graphicsBuffer.buffer, 0, 1, vertexBuffers, offsets);

			std::vector<VkDescriptorSet> sets(descriptors.size());
			int							 i = 0;
			for (auto &s : sets)
			{
				s = descriptors[i].descriptorSets;
				i++;
			}

			vkCmdBindDescriptorSets(pipeline.graphicsBuffer.buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.layout, 0,
									sets.size(), sets.data(), 0, nullptr);

			vkCmdDraw(pipeline.graphicsBuffer.buffer, vertexBuffer.vertexCount, 1, 0, 0);
		}

		void startDraw()
		{
			vkWaitForFences(instance.device, 1, &inFlightFences, VK_TRUE, UINT64_MAX);

			VkResult result = vkAcquireNextImageKHR(instance.device, window.swapChain, UINT64_MAX,
													imageAvailableSemaphores, VK_NULL_HANDLE, &imageIndex);

			if (result == VK_ERROR_OUT_OF_DATE_KHR)
			{
				recreateSwapChain();

				return;
			}
			else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
			{
				throw std::runtime_error("failed to acquire swap chain image!");
			}

			vkResetFences(instance.device, 1, &inFlightFences);

			vkResetCommandBuffer(pipeline.graphicsBuffer.buffer, 0);

			static int frame = 0;
			frame++;

			VkCommandBufferBeginInfo beginInfo{};
			beginInfo.sType			   = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			beginInfo.flags			   = 0;		  // Optional
			beginInfo.pInheritanceInfo = nullptr; // Optional

			if (vkBeginCommandBuffer(pipeline.graphicsBuffer.buffer, &beginInfo) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to begin recording command buffer!");
			}

			VkRenderPassBeginInfo renderPassInfo{};
			renderPassInfo.sType	   = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
			renderPassInfo.renderPass  = window.renderPass;
			renderPassInfo.framebuffer = window.swapChainFramebuffers[imageIndex];

			renderPassInfo.renderArea.offset = {0, 0};
			renderPassInfo.renderArea.extent = window.swapChainExtent;

			VkClearValue clearColor		   = {{{0.0f, 0.0f, 0.0f, 0.0f}}};
			renderPassInfo.clearValueCount = 1;
			renderPassInfo.pClearValues	   = &clearColor;

			vkCmdBeginRenderPass(pipeline.graphicsBuffer.buffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
		}

		void endDraw()
		{
			vkCmdEndRenderPass(pipeline.graphicsBuffer.buffer);

			if (vkEndCommandBuffer(pipeline.graphicsBuffer.buffer) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to record command buffer!");
			}

			VkSubmitInfo submitInfo{};
			submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

			VkSemaphore			 waitSemaphores[] = {imageAvailableSemaphores};
			VkPipelineStageFlags waitStages[]	  = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
			submitInfo.waitSemaphoreCount		  = 1;
			submitInfo.pWaitSemaphores			  = waitSemaphores;
			submitInfo.pWaitDstStageMask		  = waitStages;

			submitInfo.commandBufferCount = 1;
			submitInfo.pCommandBuffers	  = &pipeline.graphicsBuffer.buffer;

			VkSemaphore signalSemaphores[]	= {renderFinishedSemaphores};
			submitInfo.signalSemaphoreCount = 1;
			submitInfo.pSignalSemaphores	= signalSemaphores;

			if (vkQueueSubmit(pipeline.graphicsQueue.queue, 1, &submitInfo, inFlightFences) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to submit draw command buffer!");
			}

			VkPresentInfoKHR presentInfo{};
			presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

			presentInfo.waitSemaphoreCount = 1;
			presentInfo.pWaitSemaphores	   = signalSemaphores;

			VkSwapchainKHR swapChains[] = {window.swapChain};
			presentInfo.swapchainCount	= 1;
			presentInfo.pSwapchains		= swapChains;
			presentInfo.pImageIndices	= &imageIndex;

			presentInfo.pResults = nullptr; // optional

			VkResult result = vkQueuePresentKHR(window.presentQueue.queue, &presentInfo);

			if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || window.framebufferResized)
			{
				window.framebufferResized = false;
				recreateSwapChain();
			}
			else if (result != VK_SUCCESS)
			{
				throw std::runtime_error("failed to present swap chain image!");
			}

			currentFrame = (currentFrame + 1) % 2;
		}

		void cleanup()
		{
			cleanupSwapChain();

			vkDestroyDescriptorPool(instance.device, instance.descriptorPool, nullptr);

			instance.vmaAllocator.destroy();

			vkDestroySemaphore(instance.device, imageAvailableSemaphores, nullptr);
			vkDestroySemaphore(instance.device, renderFinishedSemaphores, nullptr);
			vkDestroyFence(instance.device, inFlightFences, nullptr);

			/*vkDestroyCommandPool(instance.device, instance.commandPool.commandPool,
			 * nullptr);*/

			/*vkDestroyPipeline(instance.device, graphicsPipeline, nullptr);*/
			/**/
			/*vkDestroyPipelineLayout(instance.device, pipelineLayout, nullptr);*/
			/**/
			/*vkDestroyRenderPass(instance.device, renderPass, nullptr);*/

			vkDestroyDevice(instance.device, nullptr);

			vkDestroySurfaceKHR(instance.instance, window.surface, nullptr);

			if (enableValidationLayers)
			{
				DestroyDebugUtilsMessengerEXT(instance.instance, instance.debugMessenger, nullptr);
			}

			vkDestroyInstance(instance.instance, nullptr);

			window.destroy();

			glfwTerminate();
		}
};

int main()
{
	HelloTriangleApplication app;

	try
	{
		app.run();
	}
	catch (const std::exception &e)
	{
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
