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
  // // Get the position in worldspace
  // vec3 grav_position = planet_positions.buf[uniforms.id].pos;

  // mat4 gravityTranslation = mat4(0.0);

  // gravityTranslation[0][0] = 1.0;
  // gravityTranslation[1][1] = 1.0;
  // gravityTranslation[2][2] = 1.0;

  // gravityTranslation[3] = vec4(grav_position, 1.0);

  // // Construct a real model matrix
  // mat4 realmodel = gravityTranslation * uniforms.modelMatrix;

  // In world-space
  interpolatedPosition = (planet_positions.buf[uniforms.id].modelMatrix * vec4(position, 1.0)).xyz;
  interpolatedNormal = normalize(mat3(transpose(inverse(planet_positions.buf[uniforms.id].modelMatrix))) * normal).xyz;

  // In object-space
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


  gl_Position = uniforms.projectionMatrix * uniforms.viewMatrix * planet_positions.buf[uniforms.id].modelMatrix * terrainScale * vec4(position, 1.0);
}