#version 450

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
} planet_data;

layout(push_constant) uniform PushConstantData {
  uint nr_of_casters;
} pc;

layout(location = 0) in vec3 interpolatedPosition;
layout(location = 1) in vec3 interpolatedNormal;
layout(location = 2) in vec3 interpolatedLocalPosition;

layout(location = 0) out vec4 frag_color;

#include <noise.glsl>
#include <lighting.glsl>

float shininess = 50.0;

void main() {
  vec3 normalPosition = normalize(interpolatedLocalPosition);

  // Calculate f by combining multiple noise layers using different density
  float granules = 1.0 - fnoise(15.0 * normalPosition, uniforms.seed, 10, 0.8);

  float darks = max(fnoise(4.0 * normalPosition, uniforms.seed, 5, 0.6), 0.0) * max(fnoise(3.0 * normalPosition, uniforms.seed, 5, 0.4), 0.0) * 2.5;

  frag_color = vec4(mix(uniforms.primaryColor, uniforms.secondaryColor, 1.0 - granules) - vec3(max(darks, 0.0)), 1.0);
}