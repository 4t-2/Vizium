#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec3 vel;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragcoord;

void main() {
    gl_Position = vec4(inPosition, 1.0);
    fragColor = inColor;
	fragcoord = inPosition.xy;
}
