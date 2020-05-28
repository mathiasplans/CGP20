#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;

layout(set = 0, binding = 0) uniform Data {
  mat4 modelMatrix;
  mat4 viewMatrix;
  mat4 projMatrix;
  float seed;
} uniforms;

layout(location = 0) out vec3 interpolatedPosition;
layout(location = 1) out vec3 interpolatedNormal;
layout(location = 2) out vec3 interpolatedLocalPosition;

void main() {
  // In world-space
  interpolatedPosition = (uniforms.modelMatrix * vec4(position, 1.0)).xyz;
  interpolatedNormal = normalize(mat3(transpose(inverse(uniforms.modelMatrix))) * normal).xyz;

  // In object-space
  interpolatedLocalPosition = position;

  gl_Position = uniforms.projMatrix * uniforms.viewMatrix * uniforms.modelMatrix * vec4(position, 1.0);
}