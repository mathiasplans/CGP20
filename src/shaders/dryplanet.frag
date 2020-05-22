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

float shininess = 250.0;

void main() {
  // Luminosity, get from texture
  vec4 lposr = texture2D(locs, vec2(0.5 / float(bodycount), 0.5));
  vec4 pposr = texture2D(locs, vec2((float(id) + 0.5) / float(bodycount), 0.5));
  float lum = luminosity(locs, id, bodycount, interpolatedPosition, lposr);

  // Distance from light
  float ldist = length(lposr.xyz - pposr.xyz);

  // Normalized local position
  vec3 normalPosition = interpolatedLocalPosition / pposr.w;

  // Calculate f by combining multiple noise layers using different density
  float f = 0.0;
  f += 1.4 * fnoise(0.5 * normalPosition, seed, 5, 0.2);
  f += 1.6 * fnoise(1.0 * normalPosition, seed, 8, 0.6);
  f += 0.2 * fnoise(2.0 * normalPosition, seed, 2, 0.5);
  f += 0.2 * fnoise(5.0 * normalPosition, seed, 2, 0.2);

  f *= 1.8;

  // Craters
  float crater = (1.0 - voronoi(normalPosition * 1.6, seed));
  crater = pow(abs(crater), 19.5);

  float c;
  if (crater > 0.001 && crater < 0.01)
    c = 1.017;

  else
    c = max(mix(1.0, 1.0 - pposr.w / 2.0, crater), 1.0 - pposr.w / 50.0);

  // Biomes
  float height = interpolatedLocalPosition.y + fnoise(15.0 * normalPosition, seed, 6, 0.45) + 3.0 * noise(1.5 * normalPosition, seed);
  float theight = (height - obliquity) / pposr.w;
  height / pposr.w;

  float iciness = abs(theight) + ldist / 8000.0 + ldist / 64000.0;

  // 1. Find normal
  vec3 n = normalize(interpolatedNormal);

  // 3. Find the direction towards the viewer, normalize.
  vec3 v = normalize(viewPosition - interpolatedPosition);

  // 4. Find the direction towards the light source, normalize.
  vec3 l = normalize(lposr.xyz - interpolatedPosition);

  // Surface colors
  vec3 noiseColor;

  // Biomes
  // Ice
  if (iciness > 0.98) {
    noiseColor = vec3(0.88, 0.9, 0.9);

    // Very minor color variation
    float icecrack = voronoi(2.7 * normalPosition, seed);
    noiseColor.xy -= vec2(icecrack / 16.0);

    float snow = abs(noise(4.0 * normalPosition, seed) / 8.0);
    noiseColor.x += snow;
  }

  // Land
  else {
    // 3-way interpolation
    float nf = (f + 1.0) / 2.0;
    float w1 = nf * nf;
    float w2 = -2.0 * (nf - 1.0) * nf;
    float w3 = (nf - 1.0) * (nf - 1.0);

    vec3 elevated = vec3(max(0.0, f));
    vec3 dust = vec3(0.0);

    if (elevated == vec3(0.0))
      dust = vec3(0.5) * max(noise(0.6 * normalPosition, seed), 0.0) / (2.0 - c);

    noiseColor = w1 * color[0] + w2 * color[1] + w3 * color[2] - dust * w3;
  }

  noiseColor = min(noiseColor, vec3(1.0));

  // Atmosphere glow
  // Get dot profuct between planet surface normal and vector to viewer
  // Then power it with a number to get it closer to the edge
  float glowIntensity = pow(1.0 - abs(dot(v, n)), 4.0);

  // Only show glow where the light is
  float glowDirection = dot(n, l);

  // Calculate the glow
  // Have to make sure that the glow is non-negative
  vec3 glow = max(colorAtm * glowIntensity * glowDirection, vec3(0.0));

  // Diffuse lighting
  float diffuse = orenNayar(l, n, v, 0.3);

  // Put Diffuse, specular and glow light together to get the end result
  vec3 interpolatedColor = lum * (noiseColor * diffuse + glow);

  gl_FragColor = vec4(interpolatedColor, 1.0);
  //gl_FragColor = vec4(vec3(icecrack), 1.0);
}