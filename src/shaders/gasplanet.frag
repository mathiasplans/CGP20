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

void main() {
    // Luminosity, get from texture
    vec4 lposr = texture2D(locs, vec2(0.5 / float(bodycount), 0.5));
    vec4 pposr = texture2D(locs, vec2((float(id) + 0.5) / float(bodycount), 0.5));
    float lum = luminosity(locs, id, bodycount, interpolatedPosition, lposr);

    vec3 n = normalize(interpolatedNormal);
    vec3 v = normalize(viewPosition - interpolatedPosition);
    vec3 l = normalize(lposr.xyz - interpolatedPosition);
    vec3 pos = interpolatedLocalPosition / size;

    // we use the y-coordinate to get effectively height-dependent 1D noise
    // this will be used to create a random angular speed vector different for each height
    // negative values is negative speed, positive values is positive speed
    float ynoise = noise(20.0 * vec3(pos.y), seed) * 20.0;
    float yrotamount = sin(time) * ynoise; // alpha = w * t

    // y-axis rotation matrix, angle scales by rotation amount and therefore changes in time
    mat3 yrot = mat3(
        cos(yrotamount)	, 0.0, sin(yrotamount),
        0.0				, 1.0, 0.0,
        -sin(yrotamount), 0.0, cos(yrotamount)
    );
    // rotate current position around x-axis
    pos = yrot * pos;

    float f = 0.0;
    f += 0.03 * (1.0 - abs(noise( 15.0 * pos, seed)));
    f += 0.01 * (1.0 - abs(noise( 21.0 * pos, seed)));
    f += 0.005 * (1.0 - abs(noise(100.0 * pos, seed)));

    // one choice is to use y-coordinate dependent noise to derive color here,
    // however I went for xz-plane symmetric appearance as saturn kind of looks like that

    // for color derivation, y-coordinate is used
    // for xz-plane symmetry, abs() is used
    float h = abs(pos.y) + f;

    vec3 color;
    if(h > 0.6){
        color = mix(colorBeige, colorDarkBeige, (h - 0.6) / 0.4);
    } else if (h > 0.3){
        color = mix(colorRed, colorBeige, (h - 0.3) / 0.3);
    } else if (h > 0.15){
        color = mix(colorGrey, colorRed, (h - 0.15) / 0.15);
    } else {
        color = mix(colorRed, colorGrey, h / 0.15);
    }

    // Diffuse lighting
    float diffuse = orenNayar(l, n, v, 0.3);
    vec3 interpolatedColor = lum * (color * diffuse);

    gl_FragColor = vec4(interpolatedColor, 1.0);
}