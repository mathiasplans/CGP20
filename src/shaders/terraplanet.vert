#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;

layout(set = 0, binding = 0) uniform Data {
  mat4 modelMatrix;
  mat4 viewMatrix;
  mat4 projectionMatrix;
  vec4 viewPosition;

  int id;
  float seed;
  float size;
  vec4 color[6];
  vec3 colorAtm;
  vec3 colorWater;
  vec3 colorDeepWater;
  float obliquity;
} uniforms;

struct planet_struct {
    vec3 pos;
    vec3 acceleration;
    float mass;
    float rad;
};

// layout(set = 0, binding = 1) buffer Planet {
//     planet_struct buf[];
// } planet_positions;

layout(location = 0) out vec3 interpolatedPosition;
layout(location = 1) out vec3 interpolatedNormal;
layout(location = 2) out vec3 interpolatedLocalPosition;

#include <noise.glsl>

void main() {
  interpolatedPosition = (uniforms.modelMatrix * vec4(position, 1.0)).xyz;
  interpolatedNormal = normalize(mat3(transpose(inverse(uniforms.modelMatrix))) * normal).xyz;
  interpolatedLocalPosition = position;

  // vec3 normalPosition = position / size;

  // float f = 0.0;
  // f += 0.4 * fnoise(0.5 * normalPosition, seed, 10, 0.7);
  // f += 0.6 * fnoise(1.0 * normalPosition, seed, 8, 0.6);
  // f += 0.7 * fnoise(2.0 * normalPosition, seed, 5, 0.2);
  // f += 0.5 * fnoise(5.0 * normalPosition, seed, 5, 0.5);
  // f += 0.1 * fnoise(8.0 * normalPosition, seed, 5, 0.8);

  // f *= 1.8;

  // if (f > 0.0)
  //   f = 1.0 + f * size / 185.0;

  // else
  //   f = 1.0;

  float f = 1.0;

  mat4 terrainScale = mat4(
      f, 0.0, 0.0, 0.0,
    0.0,   f, 0.0, 0.0,
    0.0, 0.0,   f, 0.0,
    0.0, 0.0, 0.0, 1.0
  );

  gl_Position = uniforms.projectionMatrix * uniforms.viewMatrix * uniforms.modelMatrix * terrainScale * vec4(position + planet_positions.buf[0].pos, 1.0);
}