#version 450

layout(set = 0, binding = 0) uniform Data {
  mat4 viewMatrix;
  mat4 projectionMatrix;
  vec3 viewPosition;

  uint id;
  float seed;
  float size;
  vec4 colorDeepLava;
  vec4 colorLava;
  vec4 colorBurnedGround;
  vec4 colorAsh;
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

void main() {
  // Luminosity, get from texture
  // Luminosity, get from texture
  vec3 lpos = planet_data.buf[0].pos;
  float lum = luminosity(uniforms.id, pc.nr_of_casters, interpolatedPosition);

  // Distance from light
  // float ldist = length(lposr.xyz - pposr.xyz);
  float ldist = length(lpos - planet_data.buf[uniforms.id].pos);

  vec3 normalPosition = interpolatedLocalPosition / uniforms.size;

  // Calculate f by combining multiple noise layers using different density
  float f = 0.0;
  f += 0.4 * fnoise(0.5 * normalPosition, uniforms.seed, 10, 0.7);
  f += 0.6 * fnoise(1.0 * normalPosition, uniforms.seed, 8, 0.6);
  f += 0.7 * fnoise(2.0 * normalPosition, uniforms.seed, 5, 0.2);
  f += 0.5 * fnoise(5.0 * normalPosition, uniforms.seed, 5, 0.5);
  f += 0.1 * fnoise(8.0 * normalPosition, uniforms.seed, 5, 0.8);

  f *= 1.8;

  // 1. Find normal
  vec3 n = normalize(interpolatedNormal);

  // 3. Find the direction towards the viewer, normalize.
  vec3 v = normalize(uniforms.viewPosition - interpolatedPosition);

  // 4. Find the direction towards the light source, normalize.
  vec3 l = normalize(lpos - interpolatedPosition);

  // Find angle between light and normal
  float landing = dot(l, n);
  landing = pow(max(landing, 0.0), 0.6);

  // Surface colors
  vec3 noiseColor;
  vec3 lavaglow = vec3(0.0);

  if (f > 0.4 + landing){
    noiseColor = uniforms.colorAsh.xyz;

  } else if (f > 0.2 + landing)
    noiseColor = mix(uniforms.colorBurnedGround, uniforms.colorAsh, (f - 0.2) * 5.0).xyz;

  else if (f > -0.3 + landing / 20.0)
    noiseColor = mix(uniforms.colorLava, uniforms.colorBurnedGround, (f + 0.3) * 8.5).xyz;

  else {
    float depth = min(1.0, -(f + 0.2));
    noiseColor = mix(uniforms.colorLava, uniforms.colorDeepLava, depth).xyz;
    lavaglow = noiseColor;
  }

  // Diffuse lighting
  float diffuse = orenNayar(l, n, v, 0.3);

  // Put Diffuse, specular and glow light together to get the end result
  vec3 interpolatedColor = max(lum * (noiseColor * diffuse), lavaglow);

  frag_color = vec4(interpolatedColor, 1.0);
}