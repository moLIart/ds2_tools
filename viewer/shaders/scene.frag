#version 440
layout(location=0) in vec2 albedoUv;
layout(location=1) in vec2 lmUv;
layout(location=2) in vec4 vertexColor;
layout(location=3) in vec3 surfaceNormal;

layout(std140,binding=0) uniform Frame { 
	mat4 mvp; 
	vec4 light; 
	ivec4 params; 
} frame;

layout(std140,binding=1) uniform Draw { 
	mat4 model; 
	vec4 tintGain; 
	vec4 meshCutoff; 
	ivec4 flags; 
} draw;

layout(binding=2) uniform sampler2D albedo;
layout(binding=3) uniform sampler2D lightmap;

layout(location=0) out vec4 fragColor;

// QRhi has no anisotropy sampler setting. Integrate up to 16 samples along
// the major footprint, retaining trilinear mip filtering along the minor axis.
vec4 filtered(sampler2D tex,vec2 uv,int lastMip) {
    vec2 size=vec2(textureSize(tex,0));vec2 dx=dFdx(uv),dy=dFdy(uv);
    float lx=length(dx*size),ly=length(dy*size);
    vec2 major=lx>ly?dx:dy;float large=max(lx,ly),small=max(min(lx,ly),1.0);
    int taps=int(clamp(ceil(large/small),1.0,16.0));
    float lod=clamp(log2(max(max(large/float(taps),min(lx,ly)),.00001)),0.0,float(lastMip));
    if(taps==1)return textureLod(tex,uv,lod);
    vec4 sum=vec4(0);for(int i=0;i<taps;++i)sum+=textureLod(tex,uv+major*((float(i)+.5)/float(taps)-.5),lod);
    return sum/float(taps);
}
void main() {
    if(draw.flags.y == 1){
		fragColor = vec4(draw.tintGain.rgb,1);
		return;
	}
    int mode=frame.params.x,flags=draw.flags.x;
    if(mode==4){fragColor=vec4(draw.meshCutoff.rgb,1);return;}
    vec2 uv=vec2(albedoUv.x,(flags&8)!=0?1-albedoUv.y:albedoUv.y);
    vec4 base=filtered(albedo,uv,draw.flags.z);
    if(mode==5) {
        float magnitude=length(surfaceNormal);
        vec3 n=magnitude>.00001?surfaceNormal/magnitude:vec3(0,0,1);
        if(!gl_FrontFacing)n=-n;
        fragColor=vec4(base.rgb*draw.tintGain.rgb*(.30+.70*max(dot(n,frame.light.xyz),0)),1);return;
    }
    vec2 luv=vec2(lmUv.x,(flags&16)!=0?1-lmUv.y:lmUv.y);
    bool lm=(flags&1)!=0,colors=(flags&6)==6,hasColors=(flags&4)!=0;
    vec3 lighting=vec3(1);vec3 lightColor=lm?filtered(lightmap,luv,draw.flags.w).rgb:vec3(.4);
    if(lm)lighting*=lightColor;if(colors)lighting*=vertexColor.rgb;
    float alpha=base.a*(colors?vertexColor.a:1);
    if(alpha<=draw.meshCutoff.a)discard;
    if(mode==0)fragColor=vec4(base.rgb*draw.tintGain.rgb*lighting*draw.tintGain.a,alpha);
    else if(mode==1)fragColor=vec4(base.rgb*draw.tintGain.rgb,1);
    else if(mode==2)fragColor=vec4(lightColor,1);
    else fragColor=vec4(hasColors?vertexColor.rgb:vec3(.4),1);
}
