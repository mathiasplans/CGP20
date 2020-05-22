#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;

layout(set = 0, binding = 0) uniform Data {
  mat4 viewMatrix;
  mat4 projectionMatrix;
  vec4 viewPosition;

  uint id;
  float seed;
  float size;
  vec4 color[6];
  vec3 colorAtm;
  vec3 colorWater;
  vec3 colorDeepWater;
  float obliquity;
} uniforms;

#include <types.glsl>

layout(set = 0, binding = 1) buffer Planet {
    planet_struct buf[];
} planet_positions;

layout(location = 0) out vec3 interpolatedPosition;
layout(location = 1) out vec3 interpolatedNormal;
layout(location = 2) out vec3 interpolatedLocalPosition;

#include <noise.glsl>

void main() {
  // In world-space
  interpolatedPosition = (planet_positions.buf[uniforms.id].modelMatrix * vec4(position, 1.0)).xyz;
  interpolatedNormal = normalize(mat3(transpose(inverse(planet_positions.buf[uniforms.id].modelMatrix))) * normal).xyz;

  // In object-space
  interpolatedLocalPosition = position;

  vec3 normalPosition = position / uniforms.size;

  float f = 0.0;
  f += 0.4 * fnoise(0.5 * normalPosition, uniforms.seed, 10, 0.7);
  f += 0.6 * fnoise(1.0 * normalPosition, uniforms.seed, 8, 0.6);
  f += 0.7 * fnoise(2.0 * normalPosition, uniforms.seed, 5, 0.2);
  f += 0.5 * fnoise(5.0 * normalPosition, uniforms.seed, 5, 0.5);
  f += 0.1 * fnoise(8.0 * normalPosition, uniforms.seed, 5, 0.8);

  f *= 1.8;

  if (f > 0.0)
    f = 1.0 + f * uniforms.size / 185.0;

  else
    f = 1.0;

  mat4 terrainScale = mat4(
      f, 0.0, 0.0, 0.0,
    0.0,   f, 0.0, 0.0,
    0.0, 0.0,   f, 0.0,
    0.0, 0.0, 0.0, 1.0
  );

  gl_Position = projectionMatrix * viewMatrix * planet_positions.buf[uniforms.id].modelMatrix * terrainScale * vec4(position, 1.0);
}