#define STB_IMAGE_IMPLEMENTATION
#include <viz.hpp>
#include <stb_image.h>

class Vertex
{
	public:
		glm::vec4 pos;
		glm::vec4 color;
		glm::vec4 vel;
};

int main()
{
	Instance instance;

	glfwInit();
	instance.bootstrap(true);

	Window	   window	  = instance.createWindow(800, 600, "Vizium");
	Dispatcher dispatcher = instance.createDispatcher();

	DescriptorLayout textureLayout =
		instance.createLayout(0, DescriptorLayout::Type::COMBINED_IMAGE_SAMPLER, DescriptorLayout::Stage::FRAGMENT);

	DescriptorLayout compLayout =
		instance.createLayout(0, DescriptorLayout::Type::STORAGE_BUFFER, DescriptorLayout::Stage::COMPUTE);

	Pipeline pipeline = instance.createGraphicsPipeline("vert.spv", "frag.spv",
														{textureLayout.layout, compLayout.layout}, window.renderPass);

	DescriptorPool descriptorPool = instance.createDescriptorPool();

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

	Buffer shaderStorageBuffer1 = instance.createBufferStaged(triangleData, Instance::BufferUsage::STORAGE_BUFFER);

	Buffer shaderStorageBuffer2 = instance.createBufferWrite(triangleData, Instance::BufferUsage::STORAGE_BUFFER);

	Descriptor compIn  = descriptorPool.createDescriptor(compLayout, &shaderStorageBuffer1, nullptr, nullptr);
	Descriptor compOut = descriptorPool.createDescriptor(compLayout, &shaderStorageBuffer2, nullptr, nullptr);

	Pipeline compPipeline = instance.createComputePipeline("compute.spv", {compLayout.layout, compLayout.layout});

	// load image from file
	int			 texWidth, texHeight, texChannels;
	stbi_uc		*pixels	   = stbi_load("./texture.jpg", &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
	VkDeviceSize imageSize = texWidth * texHeight * 4;

	if (!pixels)
	{
		throw std::runtime_error("failed to load texture image!");
	}

	Image	   texture			 = instance.createImage(texWidth, texHeight, pixels);
	Sampler	   sampler			 = instance.createSampler(VK_FILTER_NEAREST);
	Descriptor textureDescriptor = descriptorPool.createDescriptor(textureLayout, nullptr, &texture, &sampler);

	int frame = 0;

	bool flip = true;

	std::cout << window.swapChainImages.size() << '\n';

	while (!window.shouldClose())
	{
		std::this_thread::sleep_for(std::chrono::milliseconds(20));
		auto start = std::chrono::system_clock::now();
		glfwPollEvents();

		window.startDraw();

		static float shift = 0;

		if (glfwGetKey(window.window, GLFW_KEY_SPACE))
		{
			shift += 0.01;
		}

		std::vector desc = flip ? std::vector{compIn, compOut} : std::vector{compOut, compIn};

		dispatcher.startDispatch();
		dispatcher.dispatch(compPipeline, desc, distpa, 1, 1);
		dispatcher.endDispatch();

		window.draw(pointAmount, {textureDescriptor, flip ? compIn : compOut}, pipeline);

		window.endDraw();

		auto end = std::chrono::system_clock::now();
		frame++;

		flip = !flip;
	}

	Vertex *map;
	shaderStorageBuffer2.map((void **)&map);
	for (int i = 0; i < 3; i++)
	{
		std::cout << map[i].pos.x << " " << map[i].pos.y << " " << map[i].pos.z << " " << map[i].pos.w << "\n";
		std::cout << map[i].color.x << " " << map[i].color.y << " " << map[i].color.z << " " << map[i].color.w << "\n";
		std::cout << "\n";
	}

	vkDeviceWaitIdle(instance.device);

	instance.vmaAllocator.destroy();
	vkDestroyDevice(instance.device, nullptr);

	vkDestroySurfaceKHR(instance.instance, window.surface, nullptr);

	if (instance.enableValidationLayers)
	{
		instance.DestroyDebugUtilsMessengerEXT(instance.instance, instance.debugMessenger, nullptr);
	}

	vkDestroyInstance(instance.instance, nullptr);

	window.destroy();

	glfwTerminate();
}
