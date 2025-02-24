#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#include <viz.hpp>

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

	Window	   window	  = instance.createWindow(1800, 800, "Vizium", true);
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

	bool flip = true;

	std::cout << window.swapChainImages.size() << '\n';

	while (!window.shouldClose())
	{
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

		ImGui::ShowDemoWindow();

		static bool thing = true;
		static float fl = 0;
		{
			static float f		 = 0.0f;
			static int	 counter = 0;

			ImGui::Begin("Hello, world!"); // Create a window called "Hello, world!"
										   // and append into it.

			ImGui::Text("This is some useful text."); // Display some text (you can
													  // use a format strings too)
			ImGui::Checkbox("Demo Window",
							&thing); // Edit bools storing our window open/close state
			ImGui::Checkbox("Another Window", &thing);

			ImGui::SliderFloat("float", &fl, 0.0f,
							   1.0f); // Edit 1 float using a slider from 0.0f to 1.0f

			if (ImGui::Button("Button")) // Buttons return true when clicked (most
										 // widgets return true when edited/activated)
				counter++;
			ImGui::SameLine();
			ImGui::Text("counter = %d", counter);
			auto io = ImGui::GetIO();
			ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
			ImGui::End();
		}


		window.waitTillInterval(20);
		window.endDraw();

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
