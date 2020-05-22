#version 450

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
  // Luminosity, get from texture
  vec4 lposr = texture2D(locs, vec2(0.5 / float(bodycount), 0.5));
  vec4 pposr = texture2D(locs, vec2((0.5 + float(id)) / float(bodycount), 0.5));
  float lum = luminosity(locs, id, bodycount, interpolatedPosition, lposr);

  vec3 normalPosition = interpolatedLocalPosition / pposr.w;

  // Calculate f by combining multiple noise layers using different density
  float f = 0.0;
  f += 2.5 * fnoise(0.5 * normalPosition, seed, 10, 0.7);
  f += 2.3 * fnoise(1.0 * normalPosition, seed, 8, 0.6);
  f += 2.7 * fnoise(2.0 * normalPosition, seed, 5, 0.2);
  f += 1.5 * fnoise(5.0 * normalPosition, seed, 5, 0.5);
  f += 1.1 * fnoise(8.0 * normalPosition, seed, 5, 0.8);

  // Craters
  float crater = (1.0 - voronoi(normalPosition * 1.6, seed));
  crater = pow(crater, 19.5);

  float c;
  if (crater > 0.001 && crater < 0.01)
    c = 1.017;

  else
    c = max(mix(1.0, 1.0 - pposr.w / 2.0, crater), 1.0 - pposr.w / 50.0);

  // 1. Find normal
  vec3 n = normalize(interpolatedNormal);

  // 3. Find the direction towards the viewer, normalize.
  vec3 v = normalize(-interpolatedPosition);

  // 4. Find the direction towards the light source, normalize.
  vec3 l = normalize(lposr.xyz - interpolatedPosition);

  if (0.995 < c)
    f /= 1.5;

  vec3 noiseColor;

  // 3-way interpolation
  float nf = (f + 1.0) / 2.0;
  float w1 = nf * nf;
  float w2 = -2.0 * (nf - 1.0) * nf;
  float w3 = (nf - 1.0) * (nf - 1.0);

  vec3 dust = vec3(1.7) * max(noise(0.6 * normalPosition, seed), 0.0) / (2.0 - c);
  noiseColor = w1 * colorLightGrey + w2 * colorGrey + w3 * colorDarkGrey + dust;

  // Diffuse lighting
  float diffuse = orenNayar(l, n, v, 0.3);

  // Put Diffuse, specular and glow light together to get the end result
  vec3 interpolatedColor = lum * noiseColor * diffuse;

  gl_FragColor = vec4(interpolatedColor, 1.0);
}