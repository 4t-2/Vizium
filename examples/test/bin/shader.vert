#version 450

struct Particle {
    vec4 position;
    vec4 color;
	vec4 vel;
};

layout(std140, binding=0, set=1) buffer Data
{
Particle particle[];
};

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragcoord;

void main() {
    gl_Position = vec4(particle[gl_VertexIndex].position.xyz, 1.0);
    fragColor = particle[gl_VertexIndex].color.xyz;
	fragcoord = particle[gl_VertexIndex].position.xy;
}
