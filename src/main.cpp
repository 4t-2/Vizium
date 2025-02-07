#include <algorithm>
#include <array>
#include <chrono>
#include <glm/ext/matrix_transform.hpp>
#include <limits>
#include <map>
#include <optional>
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

#define PRINT(x) std::cout << x << '\n';

// Vizium

class Window
{
	public:
		GLFWwindow *window;
		bool		framebufferResized = false;

		void setup(int width, int height, std::string name)
		{
			glfwInit();

			glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
			glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

			window = glfwCreateWindow(width, height, name.c_str(), nullptr, nullptr);

			glfwSetWindowUserPointer(window, this);
			glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
		}

		static void framebufferResizeCallback(GLFWwindow *window, int width, int height)
		{
			((Window *)glfwGetWindowUserPointer(window))->framebufferResized = true;
		}
};

class Instance
{
};

struct Vertex
{
		glm::vec3 pos;
		glm::vec3 color;

		static VkVertexInputBindingDescription getBindingDescription()
		{
			VkVertexInputBindingDescription bindingDescription{};
			bindingDescription.binding	 = 0;
			bindingDescription.stride	 = sizeof(Vertex);
			bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

			return bindingDescription;
		}

		static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions()
		{
			std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{};

			attributeDescriptions[0].binding  = 0;
			attributeDescriptions[0].location = 0;
			attributeDescriptions[0].format	  = VK_FORMAT_R32G32B32_SFLOAT;
			attributeDescriptions[0].offset	  = offsetof(Vertex, pos);

			attributeDescriptions[1].binding  = 0;
			attributeDescriptions[1].location = 1;
			attributeDescriptions[1].format	  = VK_FORMAT_R32G32B32_SFLOAT;
			attributeDescriptions[1].offset	  = offsetof(Vertex, color);

			return attributeDescriptions;
		}
};

class VertexBuffer
{
	public:
		VkBuffer	  vertexBuffer;
		VmaAllocation vertexAllocation;
		int			  vertexCount;

		void create(VkQueue graphicsQueue, VkCommandPool commandPool, VkPhysicalDevice physicalDevice, VkDevice device,
					VkInstance instance, VmaAllocator vmaAllocator, std::vector<Vertex> vertices)
		{
			vertexCount = vertices.size();

			VkBufferCreateInfo stagingBufferInfo;
			stagingBufferInfo.sType		  = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
			stagingBufferInfo.size		  = sizeof(vertices[0]) * vertices.size();
			stagingBufferInfo.usage		  = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
			stagingBufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
			stagingBufferInfo.flags		  = 0;
			stagingBufferInfo.pNext		  = nullptr;

			VmaAllocationCreateInfo stagingAllocatorInfo = {};
			stagingAllocatorInfo.usage					 = VMA_MEMORY_USAGE_CPU_ONLY;
			stagingAllocatorInfo.flags					 = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

			VkBuffer	  stagingBuffer;
			VmaAllocation stagingAllocation;

			vmaCreateBuffer(vmaAllocator, &stagingBufferInfo, &stagingAllocatorInfo, &stagingBuffer, &stagingAllocation,
							nullptr);

			void *mappedData;
			vmaMapMemory(vmaAllocator, stagingAllocation, &mappedData);
			memcpy(mappedData, vertices.data(), sizeof(vertices[0]) * vertices.size());
			vmaUnmapMemory(vmaAllocator, stagingAllocation);

			VkBufferCreateInfo gpuBufferInfo;
			gpuBufferInfo.sType		  = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
			gpuBufferInfo.size		  = sizeof(vertices[0]) * vertices.size();
			gpuBufferInfo.usage		  = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
			gpuBufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
			gpuBufferInfo.flags		  = 0;
			gpuBufferInfo.pNext		  = nullptr;

			VmaAllocationCreateInfo gpuAllocInfo = {
				.usage = VMA_MEMORY_USAGE_GPU_ONLY // Allocates from DEVICE_LOCAL memory
			};

			vmaCreateBuffer(vmaAllocator, &gpuBufferInfo, &gpuAllocInfo, &vertexBuffer, &vertexAllocation, nullptr);

			copyBuffer(commandPool, device, graphicsQueue, stagingBuffer, vertexBuffer,
					   sizeof(vertices[0]) * vertices.size());

			vmaDestroyBuffer(vmaAllocator, stagingBuffer, stagingAllocation);
		}

		static void copyBuffer(VkCommandPool commandPool, VkDevice device, VkQueue graphicsQueue, VkBuffer srcBuffer,
							   VkBuffer dstBuffer, VkDeviceSize size)
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

			VkBufferCopy copyRegion{};
			copyRegion.srcOffset = 0; // Optional
			copyRegion.dstOffset = 0; // Optional
			copyRegion.size		 = size;
			vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

			vkEndCommandBuffer(commandBuffer);

			VkSubmitInfo submitInfo{};
			submitInfo.sType			  = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			submitInfo.commandBufferCount = 1;
			submitInfo.pCommandBuffers	  = &commandBuffer;

			vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
			vkQueueWaitIdle(graphicsQueue);

			vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
		}

		void destroy(VmaAllocator vmaAllocator)
		{
			vmaDestroyBuffer(vmaAllocator, vertexBuffer, vertexAllocation);
		}
};

struct UniformBufferObject
{
		glm::mat4 model;
		glm::mat4 view;
		glm::mat4 proj;
};

class DescriptorBuffer
{
	public:
		VkDescriptorSet descriptorSets;
		VkBuffer		uniformBuffers;
		VkDeviceMemory	uniformBuffersMemory;
		void		   *uniformBuffersMapped;
		VkDescriptorSet descriptorSets2;

		VkInstance		 &instance;
		VkDevice		 &device;
		VkPhysicalDevice &physicalDevice;

		DescriptorBuffer(VkInstance &instance, VkDevice &device, VkPhysicalDevice &physicalDevice)
			: instance(instance), device(device), physicalDevice(physicalDevice)
		{
		}

		uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties)
		{
			VkPhysicalDeviceMemoryProperties memProperties;
			vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
			for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
			{
				if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
				{
					return i;
				}
			}

			throw std::runtime_error("failed to find suitable memory type!");
		}

		void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,
						  VkBuffer &buffer, VkDeviceMemory &bufferMemory)
		{
			VkBufferCreateInfo bufferInfo{};
			bufferInfo.sType	   = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
			bufferInfo.size		   = size;
			bufferInfo.usage	   = usage;
			bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

			if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to create buffer!");
			}

			VkMemoryRequirements memRequirements;
			vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

			VkMemoryAllocateInfo allocInfo{};
			allocInfo.sType			  = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
			allocInfo.allocationSize  = memRequirements.size;
			allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

			if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to allocate buffer memory!");
			}

			vkBindBufferMemory(device, buffer, bufferMemory, 0);
		}

		void create(VkDescriptorPool descriptorPool, VkDescriptorSetLayout descriptorSetLayout,
					VkImageView textureImageView, VkSampler textureSampler)
		{
			// UNIFORM
			VkDeviceSize bufferSize = sizeof(UniformBufferObject);

			createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
						 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformBuffers,
						 uniformBuffersMemory);

			vkMapMemory(device, uniformBuffersMemory, 0, bufferSize, 0, &uniformBuffersMapped);

			// SET

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

			VkDescriptorBufferInfo bufferInfo{};
			bufferInfo.buffer = uniformBuffers;
			bufferInfo.offset = 0;
			bufferInfo.range  = sizeof(UniformBufferObject);

			VkDescriptorImageInfo imageInfo;
			imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			imageInfo.imageView	  = textureImageView;
			imageInfo.sampler	  = textureSampler;

			std::array<VkWriteDescriptorSet, 2> descriptorWrite{};
			descriptorWrite[0].sType			= VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrite[0].dstSet			= descriptorSets;
			descriptorWrite[0].dstBinding		= 0;
			descriptorWrite[0].dstArrayElement	= 0;
			descriptorWrite[0].descriptorType	= VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			descriptorWrite[0].descriptorCount	= 1;
			descriptorWrite[0].pBufferInfo		= &bufferInfo;
			descriptorWrite[0].pImageInfo		= nullptr; // Optional
			descriptorWrite[0].pTexelBufferView = nullptr; // optional

			descriptorWrite[1].sType			= VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrite[1].dstSet			= descriptorSets;
			descriptorWrite[1].dstBinding		= 1;
			descriptorWrite[1].dstArrayElement = 0;
			descriptorWrite[1].descriptorType	= VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			descriptorWrite[1].descriptorCount = 1;
			descriptorWrite[1].pImageInfo		= &imageInfo;

			vkUpdateDescriptorSets(device, descriptorWrite.size(), descriptorWrite.data(), 0, nullptr);
		}

		void destroy()
		{
			vkDestroyBuffer(device, uniformBuffers, nullptr);
			vkFreeMemory(device, uniformBuffersMemory, nullptr);
		}

		void update(UniformBufferObject ubo)
		{
			memcpy(uniformBuffersMapped, &ubo, sizeof(ubo));
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
			: descriptorBuffer(instance, device, physicalDevice), descriptorBuffer2(instance, device, physicalDevice)
		{
		}

	private:
		GLFWwindow				  *window = nullptr;
		VkInstance				   instance;
		VkDebugUtilsMessengerEXT   debugMessenger;
		VkPhysicalDevice		   physicalDevice = VK_NULL_HANDLE;
		VkDevice				   device;
		VkQueue					   graphicsQueue;
		VkSurfaceKHR			   surface;
		VkQueue					   presentQueue;
		VkSwapchainKHR			   swapChain;
		std::vector<VkImage>	   swapChainImages;
		VkFormat				   swapChainImageFormat;
		VkExtent2D				   swapChainExtent;
		std::vector<VkImageView>   swapChainImageViews;
		VkDescriptorSetLayout	   descriptorSetLayout;
		VkPipelineLayout		   pipelineLayout;
		VkRenderPass			   renderPass;
		VkPipeline				   graphicsPipeline;
		std::vector<VkFramebuffer> swapChainFramebuffers;
		VkCommandPool			   commandPool;

		VertexBuffer triangle;
		VertexBuffer square;

		VmaAllocator vmaAllocator;

		VkDescriptorPool descriptorPool;

		DescriptorBuffer descriptorBuffer;
		DescriptorBuffer descriptorBuffer2;

		VkImage				  textureImage;
		VkDeviceMemory		  textureImageMemory;
		VkImageView			  textureImageView;
		VkSampler			  textureSampler;
		VkDescriptorSetLayout textureLayout;
		VkDescriptorSet		  textureSet;

		VkCommandBuffer commandBuffers;
		VkSemaphore		imageAvailableSemaphores;
		VkSemaphore		renderFinishedSemaphores;
		VkFence			inFlightFences;

		uint32_t currentFrame = 0;

		bool framebufferResized = false;

		VkBuffer	   indexBuffer;
		VkDeviceMemory indexBufferMemory;

		const std::vector<const char *> validationLayers = {"VK_LAYER_KHRONOS_validation"};
		const std::vector<const char *> deviceExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

#ifdef NDEBUG
		const bool enableValidationLayers = false;
#else
		const bool enableValidationLayers = true;
#endif

		void initWindow()
		{
			glfwInit();
			glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
			glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
			window = glfwCreateWindow(800, 600, "Vulkan", nullptr, nullptr);

			glfwSetWindowUserPointer(window, this);
			glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
		}

		static void framebufferResizeCallback(GLFWwindow *window, int width, int height)
		{
			((HelloTriangleApplication *)glfwGetWindowUserPointer(window))->framebufferResized = true;
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

		void createInstance()
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
			if (!enableValidationLayers)
				return;

			VkDebugUtilsMessengerCreateInfoEXT createInfo{};
			populateDebugMessengerCreateInfo(createInfo);

			if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to set up debug messenger!");
			}
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

			QueueFamilyIndices indices			   = findQueueFamilies(device);
			bool			   extensionsSupported = checkDeviceExtensionSupport(device);

			bool swapChainAdequate = false;
			if (extensionsSupported)
			{
				SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
				swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
			}

			// Application can't function without geometry shaders
			if (!deviceFeatures.geometryShader || !indices.isComplete() || !swapChainAdequate)
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
				glfwGetFramebufferSize(window, &width, &height);

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

			std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

			for (const auto &extension : availableExtensions)
			{
				requiredExtensions.erase(extension.extensionName);
			}

			return requiredExtensions.empty();
		}

		struct QueueFamilyIndices
		{
				std::optional<uint32_t> graphicsFamily;
				std::optional<uint32_t> presentFamily;

				bool isComplete()
				{
					return graphicsFamily.has_value() && presentFamily.has_value();
				}
		};

		QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device)
		{
			QueueFamilyIndices indices;
			// Assign index to queue families that could be found

			uint32_t queueFamilyCount = 0;
			vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

			std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
			vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

			int i = 0;
			for (const auto &queueFamily : queueFamilies)
			{
				if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT)
				{
					indices.graphicsFamily = i;
				}

				VkBool32 presentSupport = false;
				vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);

				if (presentSupport)
				{
					indices.presentFamily = i;
				}

				i++;
			}

			return indices;
		}

		void pickPhysicalDevice()
		{
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

		void createLogicalDevice()
		{
			QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

			std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
			std::set<uint32_t> uniqueQueueFamilies = {indices.graphicsFamily.value(), indices.presentFamily.value()};

			float queuePriority = 1.0f;
			for (uint32_t queueFamily : uniqueQueueFamilies)
			{
				VkDeviceQueueCreateInfo queueCreateInfo{};
				queueCreateInfo.sType			 = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
				queueCreateInfo.queueFamilyIndex = queueFamily;
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

			vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
			vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
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

			QueueFamilyIndices indices				= findQueueFamilies(physicalDevice);
			uint32_t		   queueFamilyIndices[] = {indices.graphicsFamily.value(), indices.presentFamily.value()};

			if (indices.graphicsFamily != indices.presentFamily)
			{
				createInfo.imageSharingMode		 = VK_SHARING_MODE_CONCURRENT;
				createInfo.queueFamilyIndexCount = 2;
				createInfo.pQueueFamilyIndices	 = queueFamilyIndices;
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

			swapChainImageFormat = surfaceFormat.format;
			swapChainExtent		 = extent;
		}

		void initVulkan()
		{
			createInstance();
			setupDebugMessenger();
			createSurface();
			pickPhysicalDevice();
			createLogicalDevice();
			createSwapChain();
			createImageViews();
			createRenderPass();
			createDescriptorSetLayout();
			createGraphicsPipeline();
			createFrameBuffers();
			createCommandPool();
			createTextureImage();
			createTextureImageView();
			createTextureSampler();
			createVertexBuffer();
			createDescriptorPool();
			createDescriptorSets();
			createCommandBuffer();
			createSyncObjects();
		}

		void createTextureSampler()
		{
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

			if (vkCreateSampler(device, &samplerInfo, nullptr, &textureSampler) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to create texture sampler!");
			}
		}

		void createTextureImageView()
		{
			textureImageView = createImageView(textureImage, VK_FORMAT_R8G8B8A8_SRGB);
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
			if (vkCreateImageView(device, &viewInfo, nullptr, &imageView) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to create texture image view!");
			}

			return imageView;
		}

		void createTextureImage()
		{
			int			 texWidth, texHeight, texChannels;
			stbi_uc		*pixels	   = stbi_load("./texture.jpg", &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
			VkDeviceSize imageSize = texWidth * texHeight * 4;

			if (!pixels)
			{
				throw std::runtime_error("failed to load texture image!");
			}

			VkBuffer	   stagingBuffer;
			VkDeviceMemory stagingBufferMemory;

			createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
						 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer,
						 stagingBufferMemory);

			void *data;
			vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &data);
			memcpy(data, pixels, static_cast<size_t>(imageSize));
			vkUnmapMemory(device, stagingBufferMemory);

			stbi_image_free(pixels);

			createImage(texWidth, texHeight, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL,
						VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
						VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, textureImage, textureImageMemory);

			transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED,
								  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
			copyBufferToImage(stagingBuffer, textureImage, static_cast<uint32_t>(texWidth),
							  static_cast<uint32_t>(texHeight));

			transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
								  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

			vkDestroyBuffer(device, stagingBuffer, nullptr);
			vkFreeMemory(device, stagingBufferMemory, nullptr);
		}

		VkCommandBuffer beginSingleTimeCommands()
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

		void endSingleTimeCommands(VkCommandBuffer commandBuffer)
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

			if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to create image!");
			}

			VkMemoryRequirements memRequirements;
			vkGetImageMemoryRequirements(device, image, &memRequirements);

			VkMemoryAllocateInfo allocInfo{};
			allocInfo.sType			  = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
			allocInfo.allocationSize  = memRequirements.size;
			allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

			if (vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to allocate image memory!");
			}

			vkBindImageMemory(device, image, imageMemory, 0);
		}

		void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout)
		{
			VkCommandBuffer		 commandBuffer = beginSingleTimeCommands();
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

			endSingleTimeCommands(commandBuffer);
		}

		void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height)
		{
			VkCommandBuffer commandBuffer = beginSingleTimeCommands();

			VkBufferImageCopy region{};
			region.bufferOffset		 = 0;
			region.bufferRowLength	 = 0;
			region.bufferImageHeight = 0;

			region.imageSubresource.aspectMask	   = VK_IMAGE_ASPECT_COLOR_BIT;
			region.imageSubresource.mipLevel	   = 0;
			region.imageSubresource.baseArrayLayer = 0;
			region.imageSubresource.layerCount	   = 1;

			region.imageOffset = {0, 0, 0};
			region.imageExtent = {width, height, 1};

			vkCmdCopyBufferToImage(commandBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

			endSingleTimeCommands(commandBuffer);
		}

		void createDescriptorSets()
		{
			descriptorBuffer.create(descriptorPool, descriptorSetLayout, textureImageView, textureSampler);
			/*descriptorBuffer2.create(descriptorPool, descriptorSetLayout);*/

			/*{*/
			/*	VkDescriptorSetLayout		layouts = textureLayout;*/
			/*	VkDescriptorSetAllocateInfo allocInfo{};*/
			/*	allocInfo.sType				 =
			 * VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;*/
			/*	allocInfo.descriptorPool	 = descriptorPool;*/
			/*	allocInfo.descriptorSetCount = 1;*/
			/*	allocInfo.pSetLayouts		 = &layouts;*/
			/**/
			/*	if (vkAllocateDescriptorSets(device, &allocInfo, &textureSet) !=
			 * VK_SUCCESS)*/
			/*	{*/
			/*		throw std::runtime_error("failed to allocate descriptor
			 * sets!");*/
			/*	}*/
			/**/
			/*	VkDescriptorImageInfo imageInfo{};*/
			/*	imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;*/
			/*	imageInfo.imageView	  = textureImageView;*/
			/*	imageInfo.sampler	  = textureSampler;*/
			/**/
			/*	VkWriteDescriptorSet descriptorWrites2;*/
			/*	descriptorWrites2.sType			  =
			 * VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;*/
			/*	descriptorWrites2.dstSet		  = textureSet;*/
			/*	descriptorWrites2.dstBinding	  = 1;*/
			/*	descriptorWrites2.dstArrayElement = 0;*/
			/*	descriptorWrites2.descriptorType  =
			 * VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;*/
			/*	descriptorWrites2.descriptorCount = 1;*/
			/*	descriptorWrites2.pImageInfo	  = &imageInfo;*/
			/**/
			/*	vkUpdateDescriptorSets(device, 1, &descriptorWrites2, 0, nullptr);*/
			/*}*/
		}

		void createDescriptorPool()
		{
			VkDescriptorPoolSize poolSize[] = {
				{.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 2},
				{.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1},
			};

			VkDescriptorPoolCreateInfo poolInfo{};
			poolInfo.sType		   = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
			poolInfo.poolSizeCount = 2;
			poolInfo.pPoolSizes	   = poolSize;

			poolInfo.maxSets = 3;

			if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to create descriptor pool!");
			}
		}

		void createDescriptorSetLayout()
		{
			VkDescriptorSetLayoutBinding uboLayoutBinding{};
			uboLayoutBinding.binding			= 0;
			uboLayoutBinding.descriptorType		= VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			uboLayoutBinding.descriptorCount	= 1;
			uboLayoutBinding.stageFlags			= VK_SHADER_STAGE_VERTEX_BIT;
			uboLayoutBinding.pImmutableSamplers = nullptr; // Optional

			VkDescriptorSetLayoutBinding samplerLayoutBinding{};
			samplerLayoutBinding.binding			= 1;
			samplerLayoutBinding.descriptorCount	= 1;
			samplerLayoutBinding.descriptorType		= VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			samplerLayoutBinding.pImmutableSamplers = nullptr;
			samplerLayoutBinding.stageFlags			= VK_SHADER_STAGE_FRAGMENT_BIT;

			std::array						bindings = {uboLayoutBinding, samplerLayoutBinding};
			VkDescriptorSetLayoutCreateInfo layoutInfo{};
			layoutInfo.sType		= VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
			layoutInfo.bindingCount = bindings.size();
			layoutInfo.pBindings	= bindings.data();

			if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to create descriptor set layout!");
			}
		}

		void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,
						  VkBuffer &buffer, VkDeviceMemory &bufferMemory)
		{
			VkBufferCreateInfo bufferInfo{};
			bufferInfo.sType	   = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
			bufferInfo.size		   = size;
			bufferInfo.usage	   = usage;
			bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

			if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to create buffer!");
			}

			VkMemoryRequirements memRequirements;
			vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

			VkMemoryAllocateInfo allocInfo{};
			allocInfo.sType			  = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
			allocInfo.allocationSize  = memRequirements.size;
			allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

			if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to allocate buffer memory!");
			}

			vkBindBufferMemory(device, buffer, bufferMemory, 0);
		}

		void createVertexBuffer()
		{
			VmaAllocatorCreateInfo allocatorInfo = {};
			allocatorInfo.vulkanApiVersion		 = VK_API_VERSION_1_3;
			allocatorInfo.physicalDevice		 = physicalDevice;
			allocatorInfo.device				 = device;
			allocatorInfo.instance				 = instance;

			if (vmaCreateAllocator(&allocatorInfo, &vmaAllocator) != VK_SUCCESS)
			{
				throw std::runtime_error("Failed to create VMA allocator");
			}
			const std::vector<Vertex> triangleData = {{{0.0f, -1.f, 0}, {1.0f, 0.0f, 0.0f}},   //
													  {{0.5f, 0.5f, 0}, {0.0f, 1.0f, 0.0f}},   //
													  {{-0.5f, 0.5f, 0}, {0.0f, 0.0f, 1.0f}}}; //
			const std::vector<Vertex> squareData   = {{{0.0f, 0.0f, 0}, {0.0f, 0.0f, 0.0f}},   //
													  {{0.0f, 0.5f, 0}, {0.0f, 1.0f, 0.0f}},   //
													  {{0.5f, 0.0f, 0}, {1.0f, 0.0f, 0.0f}},   //
													  {{0.5f, 0.0f, 0}, {1.0f, 0.0f, 0.0f}},   //
													  {{0.0f, 0.5f, 0}, {0.0f, 1.0f, 0.0f}},   //
													  {{5.5f, 0.5f, 0}, {1.0f, 1.0f, 0.0f}}};  //

			triangle.create(graphicsQueue, commandPool, physicalDevice, device, instance, vmaAllocator, triangleData);
			/*square.create(graphicsQueue, commandPool, physicalDevice, device,
			 * instance, vmaAllocator, squareData);*/
		}

		uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties)
		{
			VkPhysicalDeviceMemoryProperties memProperties;
			vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
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
			glfwGetFramebufferSize(window, &width, &height);
			while (width == 0 || height == 0)
			{
				glfwGetFramebufferSize(window, &width, &height);
				glfwWaitEvents();
			}

			vkDeviceWaitIdle(device);

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

			if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores) != VK_SUCCESS ||
				vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores) != VK_SUCCESS ||
				vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to create semaphores!");
			}
		}

		void cmddraw(VertexBuffer buffer, VkCommandBuffer commandBuffer, DescriptorBuffer descriptorBuffer)
		{
			VkViewport viewport{};
			viewport.x		  = 0.0f;
			viewport.y		  = 0.0f;
			viewport.width	  = static_cast<float>(swapChainExtent.width);
			viewport.height	  = static_cast<float>(swapChainExtent.height);
			viewport.minDepth = 0.0f;
			viewport.maxDepth = 1.0f;
			vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

			VkRect2D scissor{};
			scissor.offset = {0, 0};
			scissor.extent = swapChainExtent;
			vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

			VkBuffer	 vertexBuffers[] = {buffer.vertexBuffer};
			VkDeviceSize offsets[]		 = {0};
			vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);

			vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1,
									&descriptorBuffer.descriptorSets, 0, nullptr);

			vkCmdDraw(commandBuffer, buffer.vertexCount, 1, 0, 0);
		}

		void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex)
		{
			static int frame = 0;
			frame++;

			VkCommandBufferBeginInfo beginInfo{};
			beginInfo.sType			   = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			beginInfo.flags			   = 0;		  // Optional
			beginInfo.pInheritanceInfo = nullptr; // Optional

			if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to begin recording command buffer!");
			}

			VkRenderPassBeginInfo renderPassInfo{};
			renderPassInfo.sType	   = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
			renderPassInfo.renderPass  = renderPass;
			renderPassInfo.framebuffer = swapChainFramebuffers[imageIndex];

			renderPassInfo.renderArea.offset = {0, 0};
			renderPassInfo.renderArea.extent = swapChainExtent;

			VkClearValue clearColor		   = {{{1.0f, 0.0f, 0.0f, 0.0f}}};
			renderPassInfo.clearValueCount = 1;
			renderPassInfo.pClearValues	   = &clearColor;

			vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

			vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

			{
				UniformBufferObject ubo{};
				ubo.model = glm::translate(glm::mat4(1.f), {0, -0, 0});
				ubo.model = glm::scale(ubo.model, {1, 1, 1});

				ubo.model = glm::rotate(ubo.model, frame * glm::radians(1.0f), glm::vec3(0.0f, 0.0f, 1.0f));

				ubo.view =
					glm::lookAt(glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));

				ubo.proj = glm::perspective((float)glm::radians(45.0f),
											(float)swapChainExtent.width / swapChainExtent.height, 0.1f, 100.0f);

				ubo.proj[1][1] *= -1;

				descriptorBuffer.update(ubo);
				cmddraw(triangle, commandBuffer, descriptorBuffer);

				ubo.model = glm::translate(glm::mat4(1.f), {0, 0, 0});

				ubo.model = glm::rotate(ubo.model, frame * glm::radians(1.0f), glm::vec3(0.0f, 0.0f, 1.0f));

				ubo.view = glm::lookAt(glm::vec3(0.0f, -0.0f, 20.0f), glm::vec3(0.0f, 0.0f, 0.0f),
									   glm::vec3(0.0f, 1.0f, 0.0f));

				ubo.proj = glm::perspective(glm::radians(45.0f), swapChainExtent.width / (float)swapChainExtent.height,
											0.1f, 100.0f);

				ubo.proj[1][1] *= -1;

				/*descriptorBuffer2.update(ubo);*/
				/*cmddraw(square, commandBuffer, descriptorBuffer2);*/
			}

			vkCmdEndRenderPass(commandBuffer);

			if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to record command buffer!");
			}
		}

		void createCommandBuffer()
		{
			VkCommandBufferAllocateInfo allocInfo{};
			allocInfo.sType				 = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
			allocInfo.commandPool		 = commandPool;
			allocInfo.level				 = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
			allocInfo.commandBufferCount = 1;

			if (vkAllocateCommandBuffers(device, &allocInfo, &commandBuffers) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to allocate command buffers!");
			}
		}

		void createCommandPool()
		{
			QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

			VkCommandPoolCreateInfo poolInfo{};
			poolInfo.sType			  = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
			poolInfo.flags			  = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
			poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();

			if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to create command pool!");
			}
		}

		void createFrameBuffers()
		{
			swapChainFramebuffers.resize(swapChainImageViews.size());

			for (size_t i = 0; i < swapChainImageViews.size(); i++)
			{
				VkImageView attachments[] = {swapChainImageViews[i]};

				VkFramebufferCreateInfo framebufferInfo{};
				framebufferInfo.sType			= VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
				framebufferInfo.renderPass		= renderPass;
				framebufferInfo.attachmentCount = 1;
				framebufferInfo.pAttachments	= attachments;
				framebufferInfo.width			= swapChainExtent.width;
				framebufferInfo.height			= swapChainExtent.height;
				framebufferInfo.layers			= 1;

				if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS)
				{
					throw std::runtime_error("failed to create framebuffer!");
				}
			}
		}

		void createRenderPass()
		{
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

			if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to create render pass!");
			}
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

		void createGraphicsPipeline()
		{
			auto vertShaderCode = readFile("vert.spv");
			auto fragShaderCode = readFile("frag.spv");

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
			pipelineLayoutInfo.setLayoutCount		  = 1;
			pipelineLayoutInfo.pSetLayouts			  = &descriptorSetLayout;
			pipelineLayoutInfo.pushConstantRangeCount = 0;		 // Optional
			pipelineLayoutInfo.pPushConstantRanges	  = nullptr; // Optional

			if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS)
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

			pipelineInfo.layout = pipelineLayout;

			pipelineInfo.renderPass = renderPass;
			pipelineInfo.subpass	= 0;

			pipelineInfo.basePipelineHandle = VK_NULL_HANDLE; // Optional
			pipelineInfo.basePipelineIndex	= -1;			  // Optional

			if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) !=
				VK_SUCCESS)
			{
				throw std::runtime_error("failed to create graphics pipeline!");
			}

			vkDestroyShaderModule(device, fragShaderModule, nullptr);
			vkDestroyShaderModule(device, vertShaderModule, nullptr);
		}

		void createSurface()
		{
			if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to create window surface!");
			}
		}

		void mainLoop()
		{
			while (!glfwWindowShouldClose(window))
			{
				std::this_thread::sleep_for(std::chrono::milliseconds(20));
				auto start = std::chrono::system_clock::now();
				glfwPollEvents();

				drawFrame();
				auto end = std::chrono::system_clock::now();

				/*std::cout << 1000.0 /
				 * std::chrono::duration_cast<std::chrono::milliseconds>(end -
				 * start).count() << '\n';*/
			}

			vkDeviceWaitIdle(device);
		}

		void drawFrame()
		{
			vkWaitForFences(device, 1, &inFlightFences, VK_TRUE, UINT64_MAX);

			uint32_t imageIndex;
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

			vkResetCommandBuffer(commandBuffers, 0);

			recordCommandBuffer(commandBuffers, imageIndex);

			/*updateUniformBuffer(currentFrame, false);*/

			/*recordCommandBuffer(commandBuffers[currentFrame], imageIndex);*/

			VkSubmitInfo submitInfo{};
			submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

			VkSemaphore			 waitSemaphores[] = {imageAvailableSemaphores};
			VkPipelineStageFlags waitStages[]	  = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
			submitInfo.waitSemaphoreCount		  = 1;
			submitInfo.pWaitSemaphores			  = waitSemaphores;
			submitInfo.pWaitDstStageMask		  = waitStages;

			submitInfo.commandBufferCount = 1;
			submitInfo.pCommandBuffers	  = &commandBuffers;

			VkSemaphore signalSemaphores[]	= {renderFinishedSemaphores};
			submitInfo.signalSemaphoreCount = 1;
			submitInfo.pSignalSemaphores	= signalSemaphores;

			if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to submit draw command buffer!");
			}

			VkPresentInfoKHR presentInfo{};
			presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

			presentInfo.waitSemaphoreCount = 1;
			presentInfo.pWaitSemaphores	   = signalSemaphores;

			VkSwapchainKHR swapChains[] = {swapChain};
			presentInfo.swapchainCount	= 1;
			presentInfo.pSwapchains		= swapChains;
			presentInfo.pImageIndices	= &imageIndex;

			presentInfo.pResults = nullptr; // optional

			result = vkQueuePresentKHR(presentQueue, &presentInfo);

			if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized)
			{
				framebufferResized = false;
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

			vkDestroySampler(device, textureSampler, nullptr);
			vkDestroyImageView(device, textureImageView, nullptr);
			vkDestroyImage(device, textureImage, nullptr);
			vkFreeMemory(device, textureImageMemory, nullptr);

			descriptorBuffer.destroy();
			descriptorBuffer2.destroy();

			vkDestroyDescriptorPool(device, descriptorPool, nullptr);
			vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

			triangle.destroy(vmaAllocator);
			square.destroy(vmaAllocator);

			vmaDestroyAllocator(vmaAllocator);

			vkDestroySemaphore(device, imageAvailableSemaphores, nullptr);
			vkDestroySemaphore(device, renderFinishedSemaphores, nullptr);
			vkDestroyFence(device, inFlightFences, nullptr);

			vkDestroyCommandPool(device, commandPool, nullptr);

			vkDestroyPipeline(device, graphicsPipeline, nullptr);

			vkDestroyPipelineLayout(device, pipelineLayout, nullptr);

			vkDestroyRenderPass(device, renderPass, nullptr);

			vkDestroyDevice(device, nullptr);

			vkDestroySurfaceKHR(instance, surface, nullptr);

			if (enableValidationLayers)
			{
				DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
			}

			vkDestroyInstance(instance, nullptr);

			glfwDestroyWindow(window);

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
