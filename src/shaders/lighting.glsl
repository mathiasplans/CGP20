#define M_PI 3.1415926535897932384626433832795

float lambert(in vec3 l, in vec3 n) {
  float nl = dot(n, l);

  return max(0.0, nl);
}

float orenNayar(in vec3 l, in vec3 n, in vec3 v, float r) {

  float r2 = r * r;
  float a = 1.0 - 0.5 * (r2 / (r2 + 0.33));
  float b = 0.45 * (r2 / (r2 + 0.09));

  float nl = dot(n, l);
  float nv = dot(n, v);

  float ga = dot(v - n * nv, n - n * nl);

  return max(0.0, nl) * (a + b * max(0.0, ga) * max(0.0, sqrt((1.0 - nv * nv) * (1.0 - nl * nl))) / max(nl, nv));
}

/**
 * Calculates the percentage of the lighting, when the position and radiai of light and shadow caster is known
 * The value returned is in (0, 1]. Does not take inverse square law into account. Because of that, this function
 * can be called multiple times and the results can be multiplied together to get the lighting value with
 * multiple shadow casters. When all the shadows are multiplied into a product, by dividing the product with the
 * actual distance squared
 * @param lpos Light position in world coordinates
 * @param lsize Light source radius
 * @parem cpos Shadow caster position in world coordinates
 * @param csize Shadow caster radius
 * @param fpos Fragment position is world coordinates
 */
float softShadow(vec3 lpos, float lsize, vec3 cpos, float csize, vec3 fpos) {
  // Light source
  vec3 l = lpos - fpos;
  vec3 ldir = normalize(l);
  float ldist = length(l);

  // Shadow caster
  vec3 c = cpos - fpos;
  vec3 cdir = normalize(c);
  float cdist = length(c);

  // Calculate the apparent values
  float lrad = lsize / ldist;
  float crad = csize / cdist;
  float dist = acos(dot(ldir, cdir));

  // Apparent area of light source and real are (with shadows included)
  float lapp = M_PI * pow(lrad, 2.0);
  float rapp;

  // There are three possibilities:
  // * The light source and shadow caster are not overlapping
  // * The light source and shadow caster are partly overlapping
  // * The light source and shadow caster are fully overlapping

  // The shadow caster and light source are not overlapping
  if (lrad + crad <= dist) {
    rapp = lapp;
  }

  // The shadow caster and light source are fully overlapping
  else if (abs(lrad - crad) >= dist) {
    float capp = M_PI * pow(crad, 2.0);
    rapp = max(lapp - capp, 0.0);
  }

  // There is a partial overlap between light source and shadow caster
  else {
    // Calculate the angles of the two sectors, where the points on the edge are same for
    // shadow caster and light source
    float lrad2 = pow(lrad, 2.0);
    float crad2 = pow(crad, 2.0);
    float dist2 = pow(dist, 2.0);

    float lalpha = 2.0 * acos((lrad2 + dist2 - crad2) / (2.0 * lrad * dist));
    float calpha = 2.0 * acos((crad2 + dist2 - lrad2) / (2.0 * crad * dist));

    // The overlap consists of two segments
    float lsegarea = lrad2 * (lalpha - sin(lalpha)) / 2.0;
    float csegarea = crad2 * (calpha - sin(calpha)) / 2.0;

    // TODO: What if lalpha or calpha is larger than pi / 2????

    rapp = max(lapp - (lsegarea + csegarea), 0.0);
  }

  // Normalize the value
  rapp /= lapp;

  return rapp;
}

float luminosity(uint identity, uint total, vec3 fragPosition) {
  // Luminosity, get from texture
  float lum = 1.0;
  // vec4 cposr;

  // Bodies before self
  uint i;
  for (i = 1; i < identity; ++i) {
    lum *= softShadow(planet_data.buf[0].pos, planet_data.buf[0].rad, planet_data.buf[i].pos, planet_data.buf[i].rad, fragPosition);
  }

  // Bodies after self
  for (i = identity + 1; i < total; ++i) {
    lum *= softShadow(planet_data.buf[0].pos, planet_data.buf[0].rad, planet_data.buf[i].pos, planet_data.buf[i].rad, fragPosition);
  }

  return lum;
}