#version 440
layout(location = 0) in vec2 albedoUv;
layout(location = 1) in vec2 lmUv;
layout(location = 2) in vec4 vertexColor;
layout(location = 3) in vec3 surfaceNormal;

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
// flags.y: 1 solid tintGain, 2 vertex colors for gizmos; both disable skinning.
// flags.z/w: albedo/lightmap last mip indices; not used by this shader.
layout(std140, binding = 1) uniform Draw {
  mat4 model;
  vec4 tintGain;
  vec4 meshCutoff;
  ivec4 flags;
}

draw;

layout(binding = 2) uniform sampler2D albedo;
layout(binding = 3) uniform sampler2D lightmap;

layout(location = 0) out vec4 fragColor;

void main() {
  if (draw.flags.y != 0) {
    fragColor = draw.flags.y == 2 ? vertexColor : vec4(draw.tintGain.rgb, draw.tintGain.a);
    return;
  }

  int mode = frame.params.x, flags = draw.flags.x;
  if (mode == 4) {
    fragColor = vec4(draw.meshCutoff.rgb, 1);
    return;
  }

  vec2 uv   = vec2(albedoUv.x, (flags & 8) != 0 ? 1 - albedoUv.y : albedoUv.y);
  vec4 base = texture(albedo, uv);
  if (mode == 5) {
    float magnitude = length(surfaceNormal);
    vec3 n          = magnitude > .00001 ? surfaceNormal / magnitude : vec3(0, 0, 1);
    if (!gl_FrontFacing) {
      n = -n;
    }

    fragColor = vec4(base.rgb * draw.tintGain.rgb * (.30 + .70 * max(dot(n, frame.light.xyz), 0)), 1);
    return;
  }

  vec2 luv = vec2(lmUv.x, (flags & 16) != 0 ? 1 - lmUv.y : lmUv.y);
  bool lm = (flags & 1) != 0, colors = (flags & 6) == 6, hasColors = (flags & 4) != 0;
  vec3 lighting   = vec3(1);
  vec3 lightColor = lm ? texture(lightmap, luv).rgb : vec3(.4);
  if (lm) {
    lighting *= lightColor;
  }

  if (colors) {
    lighting *= vertexColor.rgb;
  }

  float alpha = base.a * (colors ? vertexColor.a : 1);
  if (alpha <= draw.meshCutoff.a) {
    discard;
  }

  if (mode == 0) {
    fragColor = vec4(base.rgb * draw.tintGain.rgb * lighting * draw.tintGain.a, alpha);
  } else if (mode == 1) {
    fragColor = vec4(base.rgb * draw.tintGain.rgb, 1);
  } else if (mode == 2) {
    fragColor = vec4(lightColor, 1);
  } else {
    fragColor = vec4(hasColors ? vertexColor.rgb : vec3(.4), 1);
  }
}
