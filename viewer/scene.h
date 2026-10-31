#pragma once

#include <QMatrix4x4>
#include <QStringList>
#include <QVector2D>
#include <QVector3D>
#include <QVector4D>
#include <QVector>
#include <cstdint>
#include <memory>

namespace ds2 {
struct Bounds {
    QVector3D low, high;
};
struct Vertex {
    float position[3] = {};
    float normal[3] = {0, 0, 1};
    float uv[2] = {};
    float lightUv[2] = {};
    float color[4] = {1, 1, 1, 1};
    uint32_t influenceStart = 0, influenceCount = 0;
};
static_assert(sizeof(Vertex) == 64);
struct Geometry {
    QVector<Vertex> vertices;
    bool hasColors = false;
};
enum class Blend { None, Alpha, Add, Modulate, Replace };
struct Material {
    QString name;
    QString albedo = "$missing$", lightmap = "$white$";
    QVector3D tint{1, 1, 1};
    Blend blend = Blend::None;
    float sort = 0, alphaCutoff = -1, gain = 1;
    bool cull = false, depthTest = true, depthWrite = true, depthOffset = false;
    bool clamp = false, useLightmap = false, useColors = false;
    bool flipAlbedo = false, flipLightmap = false;
};
struct Mesh {
    QString name;
    int geometry = 0;
    int sortGeometry = 0;
    QVector<uint32_t> indices; // triangle list, local to Geometry
    Material material;
    QMatrix4x4 transform;
    Bounds bounds; // world-space for maps
};
struct VisNode {
    Bounds bounds;
    QVector<int> children;
    QVector<int> meshes;
};
class Animation {
  public:
    virtual ~Animation() = default;
    bool playing = false, loop = true;
    double speed = 1;
    virtual int clipCount() const = 0;
    virtual QString clipName(int index) const = 0;
    virtual int clipIndex() const = 0; // -1: bind pose / stored geometry
    virtual double time() const = 0;
    virtual double duration() const = 0;
    virtual void select(int index) = 0;
    virtual void seek(double seconds) = 0;
    virtual bool update(double seconds) = 0;
    virtual const QVector<QMatrix4x4> &matrices() const = 0;
    virtual Bounds bounds() const = 0;
};
struct Scene {
    QString path, name, version, note;
    QStringList warnings;
    QVector<Geometry> geometries;
    QVector<Mesh> meshes;
    QVector<VisNode> visibility;
    QVector<QVector2D> influences; // matrix index, raw weight; no normalization
    std::shared_ptr<Animation> animation;
    Bounds bounds;
    bool map = false, sampledAnimation = false;
    int boneCount = 0;
};
Scene loadMap(const QString &path, const QString &gameRoot);
Scene loadModel(const QString &path, const QString &gameRoot);
void visibleMeshes(const Scene &scene, const QMatrix4x4 &zeroToOneClip, QVector<int> &result);
} // namespace ds2
