#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;

layout(set = 0, binding = 0) uniform Data {
  mat4 viewMatrix;
  mat4 projectionMatrix;
  vec3 viewPosition;

  uint id;
  float seed;
  float size;
  vec3 primaryColor;
  vec3 secondaryColor;
  float obliquity;
} uniforms;

#include <types.glsl>

layout(set = 0, binding = 1) buffer Planet {
    planet_struct buf[];
} planet_positions;

layout(location = 0) out vec3 interpolatedPosition;
layout(location = 1) out vec3 interpolatedNormal;
layout(location = 2) out vec3 interpolatedLocalPosition;

void main() {
  // In world-space
  interpolatedPosition = (planet_positions.buf[uniforms.id].modelMatrix * vec4(position, 1.0)).xyz;
  interpolatedNormal = normalize(mat3(transpose(inverse(planet_positions.buf[uniforms.id].modelMatrix))) * normal).xyz;

  // In object-space
  interpolatedLocalPosition = position;

  gl_Position = uniforms.projectionMatrix * uniforms.viewMatrix * planet_positions.buf[uniforms.id].modelMatrix * vec4(position, 1.0);
}