float F = 1.0 / 3.0;
float G = 1.0 / 6.0;

#extension GL_EXT_control_flow_attributes : require

// Source: https://www.shadertoy.com/view/Xsl3Dl
// Has been modified for our needs
vec3 hash(ivec3 internal, ivec3 s, float seed) {
  vec3 p = vec3(internal) + vec3(s);
  p = vec3(
    dot(p, vec3(127.1, 311.7, 74.7)),
    dot(p, vec3(269.5, 163.3, 226.1)),
    dot(p, vec3(113.5, 271.9, 124.6))
  );

  return normalize(-1.0 + 2.0 * fract(sin(p) * 43758.5453123 + seed));
}

float noise(in vec3 p, in float seed) {
  float strech = p.x + p.y + p.z;

  float skewFactor = strech * F;

  // Place input coordinates on 
  ivec3 internal = ivec3(floor(p + skewFactor));

  int squish = internal.x + internal.y + internal.z;
  float unskewFactor = float(squish) * G;

  // Unskew
  vec3 unskew = vec3(internal) - unskewFactor;

  // Distance
  vec3 dst = p - unskew;

  // Determine in which simplices we are in
  ivec3 s[4];
  s[0] = ivec3(0);
  s[3] = ivec3(1);

  if (dst.x >= dst.y) {
    if (dst.y >= dst.z) {
      s[1] = ivec3(1, 0, 0);
      s[2] = ivec3(1, 1, 0);
    }

    else if (dst.x >= dst.z) {
      s[1] = ivec3(1, 0, 0);
      s[2] = ivec3(1, 0, 1);
    }

    else {
      s[1] = ivec3(0, 0, 1);
      s[2] = ivec3(1, 0, 1);
    }
  }

  else { // if dst.x < dst.y
    if (dst.y < dst.z) {
      s[1] = ivec3(0, 0, 1);
      s[2] = ivec3(0, 1, 1);
    }

    else if (dst.x < dst.z) {
      s[1] = ivec3(0, 1, 0);
      s[2] = ivec3(0, 1, 1);
    }

    else {
      s[1] = ivec3(0, 1, 0);
      s[2] = ivec3(1, 1, 0);
    }
  }

  // Offsets for conrners
  vec3 offset[] = vec3[](
    dst,
    dst - vec3(s[1]) + G,
    dst - vec3(s[2]) + 2.0 * G,
    dst - vec3(s[3]) + 3.0 * G
  );

  // Gradients
  vec3 g[4];
  for (int i = 0; i < 4; ++i) {
    g[i] = hash(internal, s[i], seed);
  }

  // Interpolate
  vec3 tmp;
  float t = 0.0;
  float n[] = float[](0.0, 0.0, 0.0, 0.0);

  [[unroll]]
  for (int i = 0; i < 4; ++i) {
    t = 0.6 - dot(offset[i], offset[i]);

    // Get out, multiply with max/min/clamp max(0, t)
    if (t > 0.0) {
      t *= t;
      n[i] = t * t * dot(g[i], offset[i]);
    }
  }

  return (32.0 * (n[0] + n[1] + n[2] + n[3]));
}

// Source: https://www.seedofandromeda.com/blogs/49-procedural-gas-giant-rendering-with-gpu-noise
float fnoise(in vec3 p, in float seed, int octaves, float persistence) {
  // Total value so far
  float total = 0.0;

  // Accumulates highest theoretical amplitude
  float maxAmplitude = 0.0;

  float amplitude = 1.0;
  for (int i = 0; i < octaves; i++) {
    // Get the noise sample
    total += noise(p, seed) * amplitude;

    // Make the wavelength twice as small
    p *= 2.0;

    // Add to our maximum possible amplitude
    maxAmplitude += amplitude;

    // Reduce amplitude according to persistence for the next octave
    amplitude *= persistence;
  }

  // Scale the result by the maximum amplitude
  return total / maxAmplitude;
}

// Source: https://www.ronja-tutorials.com/2018/09/29/voronoi-noise.html#3d-voronoi-1
float voronoi(in vec3 p, in float seed){
  vec3 baseCell = floor(p);

  // first pass to find the closest cell
  float minDistToCell = 10.0;
  vec3 toClosestCell;
  vec3 closestCell;
  for (int x1 = -1; x1 <= 1; ++x1) {
    for (int y1 = -1; y1 <= 1; ++y1) {
      for (int z1 = -1; z1 <= 1; ++z1) {
        vec3 cell = baseCell + vec3(x1, y1, z1);
        vec3 cellPosition = cell + noise(cell, seed);
        vec3 toCell = cellPosition - p;
        float distToCell = length(toCell);
        if (distToCell < minDistToCell) {
          minDistToCell = distToCell;
          closestCell = cell;
          toClosestCell = toCell;
        }
      }
    }
  }

  return minDistToCell;
}