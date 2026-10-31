#version 440
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;
layout(location = 3) in vec2 lightUv;
layout(location = 4) in vec4 color;
layout(location = 5) in uvec2 influenceRange;

// params.x: 0 combined, 1 albedo, 2 lightmap, 3 vertex colors, 4 mesh colors, 5 model.
layout(std140, binding = 0) uniform Frame {
  mat4 mvp;
  vec4 light;
  ivec4 params;
}

frame;

// Per-mesh DrawUniform from render_widget.cpp:
// tintGain.rgb: color3 tint; tintGain.a: lighting gain, or alpha for solid draws.
// meshCutoff.rgb: mesh debug color; meshCutoff.a: alpha-test cutoff.
// flags.x bits: 1 use lightmap, 2 use vertex colors, 4 vertex colors present,
//               8 flip albedo V, 16 flip lightmap V. Bits 2 and 4 must both be set
//               for vertex colors to affect lighting and alpha.
// flags.y: 1 draws solid tintGain.rgb and disables skinning (used by the grid).
// flags.z/w: albedo/lightmap last mip indices; not used by this shader.
layout(std140, binding = 1) uniform Draw {
  mat4 model;
  vec4 tintGain;
  vec4 meshCutoff;
  ivec4 flags;
}

draw;
layout(binding = 4) uniform sampler2D influences;
layout(binding = 5) uniform sampler2D palette;
layout(location = 0) out vec2 albedoUv;
layout(location = 1) out vec2 lmUv;
layout(location = 2) out vec4 vertexColor;
layout(location = 3) out vec3 surfaceNormal;

vec4 fetchData(sampler2D tex, int i) {
  int w = textureSize(tex, 0).x;
  return texelFetch(tex, ivec2(i % w, i / w), 0);
}

void main() {
  vec4 p = vec4(position, 1);
  vec3 n = normal;
  if (frame.params.x == 5 && influenceRange.y != 0u && draw.flags.y == 0) {
    mat4 skin = mat4(0);
    for (uint i = 0u; i < influenceRange.y; ++i) {
      vec2 influence = fetchData(influences, int(influenceRange.x + i)).rg;
      int j          = int(influence.x) * 4;
      skin += mat4(fetchData(palette, j), fetchData(palette, j + 1), fetchData(palette, j + 2), fetchData(palette, j + 3)) * influence.y;
    }

    p = vec4((skin * p).xyz, 1);
    n = mat3(skin) * n;
  }

  gl_Position   = frame.mvp * draw.model * p;
  albedoUv      = uv;
  lmUv          = lightUv;
  vertexColor   = color;
  surfaceNormal = n;
}
