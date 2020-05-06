#version 450

layout(set = 0, binding = 0) uniform Data {
  mat4 modelMatrix;
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

    // sampler2D locs;
    // vec3 viewPosition;

    // positive when top half is closer to light, negative if bottom half is closer to light
  float obliquity;
} uniforms;

struct planet_struct {
    vec3 pos;
    vec3 velocity;
    float mass;
    float rad;
};

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
  // vec4 lposr = texture2D(locs, vec2(0.5 / float(bodycount), 0.5));
  vec3 lpos = 20.0 * vec3(0.0, 5.0, 10.0);
  // vec4 pposr = texture2D(locs, vec2((float(id) + 0.5) / float(bodycount), 0.5));
  vec4 pposr = vec4(0.0, 0.0, 0.0, uniforms.size);

  float lum = luminosity(uniforms.id, pc.nr_of_casters, interpolatedPosition, lpos, 2.0);

  // Distance from light
  // float ldist = length(lposr.xyz - pposr.xyz);
  float ldist = 10.0;

  // Normalized local position
  // vec3 normalPosition = interpolatedLocalPosition / pposr.w;
  vec3 normalPosition = interpolatedLocalPosition / uniforms.size;

  // Calculate f by combining multiple noise layers using different density
  float f = 0.0;
  // f += 0.4 * fnoise(0.5 * normalPosition, uniforms.seed, 10, 0.7);
  // f += 0.6 * fnoise(1.0 * normalPosition, uniforms.seed, 8, 0.6);
  // f += 0.7 * fnoise(2.0 * normalPosition, uniforms.seed, 5, 0.2);
  // f += 0.5 * fnoise(5.0 * normalPosition, uniforms.seed, 5, 0.5);
  // f += 0.1 * fnoise(8.0 * normalPosition, uniforms.seed, 5, 0.8);

  // f *= 1.8;

  f += 2.0 * fnoise(0.5 * normalPosition, uniforms.seed, 5, 0.7);
  // f = noise3(5.0 * interpolatedLocalPosition).x;

  // Biomes
  float height = interpolatedPosition.y + fnoise(15.0 * normalPosition, uniforms.seed, 6, 0.45) + 3.0 * noise(1.5 * normalPosition, uniforms.seed);
  float theight = (height - uniforms.obliquity) / pposr.w;
  height / pposr.w;

  float iciness = abs(theight) + max(f, 0.005) * ldist / 800.0 + ldist / 6400.0;

  // 1. Find normal
  vec3 n = normalize(interpolatedNormal);

  // 3. Find the direction towards the viewer, normalize.
  vec3 v = normalize(uniforms.viewPosition.xyz - interpolatedPosition);

  // 4. Find the direction towards the light source, normalize.
  vec3 l = normalize(lpos - interpolatedPosition);

  // 5. Blinn: Find the half-angle vector h
  vec3 h = normalize(l + v);

  // Surface colors, specular highlight
  float specular = 0.0;
  vec3 noiseColor;

  // Biomes
  // Ice
  // if (iciness > 0.95) {
  //   noiseColor = vec3(0.93, 1.0, 1.0);

  //   // Very minor color variation
  //   float icecrack = voronoi(2.7 * normalPosition, uniforms.seed);
  //   noiseColor.xy -= vec2(icecrack / 16.0);

  //   float snow = abs(noise(4.0 * normalPosition, uniforms.seed) / 8.0);
  //   noiseColor.x += snow;
  // }

  // Water
  /*else*/ if (f <= 0.0) {
    f = f * f * f;
    noiseColor = mix(uniforms.colorWater, uniforms.colorDeepWater, min(1.0, -f));
    specular = pow(max(0.0, dot(n, h)), 3.0 * shininess);
  }

  // // // Hot
  // // else if (abs(height) + ldist / 800.0 + fnoise(32.2 * normalPosition, uniforms.seed, 4, 0.85) < 2.0 && f > 0.02 + ldist / 3200.0 + height * height / 80.0) {
  // //   if (f > 0.3)
  // //     noiseColor = mix(uniforms.color[4], uniforms.color[5], min(1.0, (f - 0.3))).xyz;

  // //   else
  // //     noiseColor = mix(uniforms.color[3], uniforms.color[4], (f - 0.1) / 0.4).xyz;
  // // }

  // Temperate
  else {
    if (f > 0.5)
      noiseColor = mix(uniforms.color[2], uniforms.color[1], min(1.0, (f - 0.5))).xyz;

    else
      noiseColor = mix(uniforms.color[0], uniforms.color[1], (f - 0.1) / 0.4).xyz;

    // Make planets further from light less vegetationy
    noiseColor.x += sqrt(ldist) / 150.0;
    noiseColor.z += sqrt(ldist) / 400.0;
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
  vec3 glow = max(uniforms.colorAtm * glowIntensity * glowDirection, vec3(0.0));

  // Diffuse lighting
  float diffuse = orenNayar(l, n, v, 0.3);

  // Put Diffuse, specular and glow light together to get the end result
  vec3 interpolatedColor = lum * (noiseColor * diffuse + specular + glow);

  frag_color = vec4(interpolatedColor, 1.0);
  // frag_color = vec4(vec3(diffuse), 1.0);

  // if (diffuse <= 0.0)
  //   frag_color = vec4(1.0, 0.0, 0.0, 1.0);

  // if (isnan(diffuse))
  //   frag_color = vec4(0.0, 1.0, 0.0, 1.0);

  // if (isinf(diffuse))
  //   frag_color = vec4(0.0, 0.0, 1.0, 1.0);
}