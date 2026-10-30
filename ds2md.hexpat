#pragma description You Are Empty DS2MD binary model (0.9 / 1.0)
#pragma endian little
#pragma pattern_limit 8388608
#pragma array_limit 1048576

import std.io;
import std.mem;
import std.sys;

// Recovered from ds2render.dll (PE32 preferred base 0x10000000).
// SHA256: 216c22891726e624787918396112127f157de0255e70994be01fe509a3624c78
// Main reader: 0x100BC320; mesh: 0x100C2D60; skeleton: 0x100CD2C0.
// Packed fields: no alignment padding. ASCII 0.5 and binary 0.6 are not supported.

struct DS2Float3 { float v[3]; };
struct DS2Float2 { float v[2]; };
struct DS2Quaternion { float x; float y; float z; float w; };
struct DS2Matrix4 {
    float values[16]; // Loader 0x10058780 transposes these 64 bytes into memory.
};

struct DS2String {
    u16 length;
    char value[length]; // Byte length; no terminating NUL on disk.
};

struct DS2IndexData {
    u32 count;
    u16 buffer[count];
};

struct DS2BoneWeight {
    u16 bone_index;
    float weight;
};

struct DS2VertexAssignment {
    u8 bone_count;
    // Engine rejects >= 10 influences (0x100C32C4).
    std::assert(bone_count < 10, "Engine does not support >= 10 vertex influences");
    DS2BoneWeight weights[bone_count];
};

struct DS2VertexData {
    u16 count;
    DS2Float3 positions[count];
    DS2Float3 normals[count];
    DS2Float2 uv_coords[count];
    u8 influence_mode; // 0: NO assignments; nonzero: one record per vertex.
    if (influence_mode != 0) {
        DS2VertexAssignment assignments[count];
    }
};

struct DS2TriangleListVertexData<auto indices> : DS2VertexData {
} [[hex::visualize("3d", positions, indices)]];

struct DS2Mesh {
    DS2String material; // Material/shader descriptor name, not a texture.
    DS2String diffuse_texture;
    DS2String bump_texture; // Also called normalmap by the Lua override.
    DS2String luminosity_texture;
    DS2String reflection_texture;
    DS2String specular_texture;

    // Exactly the 33 bytes previously left unknown.
    DS2Float3 bounds_min;
    DS2Float3 bounds_max;
    u32 primitive_count;
    u8 triangle_strip; // 0: triangle list; nonzero: triangle strip.
    u32 skinning_count; // Engine: 0 unrigged, 1 rigid, >1 skinned; exact count meaning unconfirmed.

    DS2IndexData index_data;
    if (triangle_strip == 0) {
        DS2TriangleListVertexData<index_data.buffer> vertex_data;
    } else {
        // ImHex's 3D visualizer expects triangle lists, not strips.
        DS2VertexData vertex_data;
    }
    // Preview uses stored positions only: no rest-pose or animation skinning.
};

enum DS2CollisionType : u8 {
    Sphere = 0,
    Box = 1,
    Cylinder = 2,
    Dummy = 3
};

struct DS2CollisionObject {
    DS2String name;
    DS2CollisionType type;
    u16 bone_index; // 0xFFFF: not attached to a bone.
    DS2Matrix4 local_matrix;
    if (type == DS2CollisionType::Sphere) {
        float radius;
    } else if (type == DS2CollisionType::Box) {
        DS2Float3 lengths; // Full side lengths, not half-extents.
    } else if (type == DS2CollisionType::Cylinder) {
        float radius;
        float length;
    }
    // Type 3 (dummy), and other values in the current loader: NO parameters.
    // Reader 0x100B6EA0; ODE consumers in sv_game.dll 0x1007B404-0x1007B4A2.
};

struct DS2Bone {
    DS2String name;
    DS2Matrix4 local_rest_matrix;
    u16 parent_index; // 0xFFFF for a root.
    u16 child_count;
    u16 child_indices[child_count];
    u16 animation_slot_index; // Animation controller slot, NOT a shader slot.
};

struct DS2AnimationKey {
    // Packed size 45 bytes; reader 0x100CF790. Right key governs interpolation:
    // 0x01 hold rotation, 0x02 hold translation, 0x04 hold scale, 0x08 use scale.
    // Unidentified upper bits are preserved.
    u8 flags;
    DS2Quaternion rotation;
    DS2Float3 translation;
    DS2Float3 scale;
    float time_seconds;
};

struct DS2AnimationChannel {
    u16 bone_index;
    DS2String bone_name;
    u32 key_count;
    DS2AnimationKey keys[key_count];
};

struct DS2AnimationEvent {
    float time_seconds;
    DS2String name;
};

struct DS2Animation<auto has_events> {
    DS2String name;
    float duration_seconds;
    float time_parameter_seconds; // Probably sample interval; meaning unconfirmed.
    u16 channel_count;
    DS2AnimationChannel channels[channel_count];
    // strcmp(version, "1.0") at 0x100D145D: 0.9 omits even the count.
    if (has_events) {
        u16 event_count;
        DS2AnimationEvent events[event_count];
    }
};

struct DS2AnimationGroup {
    DS2String name;
    u16 bone_count;
    u16 bone_indices[bone_count];
    u16 animation_slot_count;
    u16 animation_slot_indices[animation_slot_count];
};

struct DS2Skeleton<auto has_events> {
    u16 bone_count;
    DS2Bone bones[bone_count];
    u16 root_count;
    u16 root_indices[root_count];
    u16 animation_count;
    DS2Animation<has_events> animations[animation_count];
    u16 group_count;
    DS2AnimationGroup groups[group_count];
    u16 animation_slot_count;
};

struct DS2Model {
    DS2String file_id;
    std::assert(file_id.value == "DS2ModelFile_1", "Not a binary DS2MD model");
    DS2String version;
    std::assert(version.value == "0.9" || version.value == "1.0", "Only binary 0.9 and 1.0 are supported");
    DS2String name;
    u8 mesh_count;
    DS2Mesh meshes[mesh_count];
    u8 collision_object_count;
    DS2CollisionObject collision_objects[collision_object_count];
    u8 has_skeleton;
    if (has_skeleton != 0) {
        // ALL bones, animations, groups and final slot count are conditional.
        DS2Skeleton<version.value == "1.0"> skeleton;
    }
};

std::assert(std::mem::read_unsigned(0, 2) == 14, "Expected binary DS2MD (not ASCII 0.5)");
DS2Model model @ 0x0;
std::assert($ == std::mem::size(), "Unparsed trailing bytes");
std::print("DS2MD {}: {} meshes, {} collision objects; consumed {} / {} bytes", model.version.value, model.mesh_count, model.collision_object_count, $, std::mem::size());
