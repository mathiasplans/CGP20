#version 450

layout(set = 0, binding = 0) uniform Data {
  mat4 modelMateix;
  mat4 viewMatrix;
  mat4 projMatrix;
  float seed;
} uniforms;

layout(location = 0) in vec3 interpolatedPosition;
layout(location = 1) in vec3 interpolatedNormal;
layout(location = 2) in vec3 interpolatedLocalPosition;

layout(location = 0) out vec4 frag_color;

#include <noise.glsl>

void main() {
  float f = 0.0;

  f += noise(interpolatedLocalPosition * 3.0, uniforms.seed);
  f *= noise(interpolatedLocalPosition * 4.0, uniforms.seed);
  f *= noise(interpolatedLocalPosition * 5.0, uniforms.seed);
  f *= noise(interpolatedLocalPosition * 17.0, uniforms.seed);

  if (f > 0.065)
    f = 1.0;
  else
    f = 0.0;

  frag_color = vec4(
    vec3(f),
    1.0
  );
}