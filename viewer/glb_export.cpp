#include "glb_export.h"
#include "texture.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <QBuffer>
#include <QDir>
#include <QFileInfo>
#include <QHash>
#include <QImage>
#include <QJsonDocument>
#include <QSaveFile>
#include <QtEndian>
#include <stdexcept>
#include <vector>

namespace ds2 {
  namespace {
    constexpr int arrayBuffer = 34962, elementArrayBuffer = 34963;
    constexpr quint64 maxGlbSize = std::numeric_limits<quint32>::max();

    [[noreturn]] void fail(const QString& message) {
      throw std::runtime_error(message.toStdString());
    }

    void warn(glb::Document& document, const QString& message) {
      if (!document.warnings.contains(message)) {
        document.warnings.append(message);
      }
    }

    int components(const char* type) {
      if (std::strcmp(type, "SCALAR") == 0) {
        return 1;
      }

      if (std::strcmp(type, "VEC2") == 0) {
        return 2;
      }

      if (std::strcmp(type, "VEC3") == 0) {
        return 3;
      }

      if (std::strcmp(type, "VEC4") == 0) {
        return 4;
      }

      if (std::strcmp(type, "MAT4") == 0) {
        return 16;
      }

      fail("Unsupported GLB accessor type");
    }

    qsizetype reserveBinary(glb::Document& document, quint64 length) {
      const quint64 offset = (quint64(document.binary.size()) + 3) & ~quint64(3);
      if (!length || length > maxGlbSize || offset > maxGlbSize - length || offset + length > quint64(std::numeric_limits<qsizetype>::max())) {
        fail("GLB binary data exceeds its supported size or is empty");
      }

      const auto oldSize = document.binary.size();
      document.binary.resize(qsizetype(offset + length));
      std::fill(document.binary.begin() + oldSize, document.binary.begin() + qsizetype(offset), char(0));
      return qsizetype(offset);
    }

    int bufferView(glb::Document& document, qsizetype offset, quint64 length, int target) {
      QJsonObject view{{"buffer", 0}, {"byteOffset", double(offset)}, {"byteLength", double(length)}};
      if (target) {
        view.insert("target", target);
      }

      const int index = int(document.bufferViews.size());
      document.bufferViews.append(view);
      return index;
    }

    int accessor(glb::Document& document, int view, int componentType, quint64 count, const char* type) {
      if (!count || count > maxGlbSize) {
        fail("Invalid GLB accessor count");
      }

      const int index = int(document.accessors.size());
      document.accessors.append(QJsonObject{{"bufferView", view}, {"componentType", componentType}, {"count", double(count)}, {"type", QString::fromLatin1(type)}});
      return index;
    }

    template <typename T>
    int integerAccessor(glb::Document& document, std::span<const T> values, const char* type, int componentType, int target) {
      const int width = components(type);
      if (values.size() % size_t(width) || values.size() > maxGlbSize / sizeof(T)) {
        fail("Invalid GLB integer accessor size");
      }

      const auto length = quint64(values.size()) * sizeof(T);
      const auto offset = reserveBinary(document, length);
      char* destination = document.binary.data() + offset;
      for (size_t i = 0; i < values.size(); ++i) {
        qToLittleEndian<T>(values[i], destination + i * sizeof(T));
      }

      return accessor(document, bufferView(document, offset, length, target), componentType, values.size() / width, type);
    }

    QString canonicalPath(const QString& path) {
      const QFileInfo info(path);
      QString result = info.canonicalFilePath();
      if (result.isEmpty()) {
        const QString parent = info.dir().canonicalPath();
        result               = parent.isEmpty() ? info.absoluteFilePath() : QDir(parent).filePath(info.fileName());
      }

      result = QDir::cleanPath(QDir::fromNativeSeparators(result));
#ifdef Q_OS_WIN
      result = result.toCaseFolded();
#endif
      return result;
    }

    QJsonArray matrix(const QMatrix4x4& value) {
      QJsonArray result;
      for (int i = 0; i < 16; ++i) {
        const float component = value.constData()[i];
        if (!std::isfinite(component)) {
          fail("Cannot export a non-finite mesh transform");
        }

        result.append(component);
      }

      return result;
    }

    class MaterialWriter {
      glb::Document& document;
      QHash<QString, int> imageCache;
      QHash<quint64, int> textureCache;
      std::array<int, 2> samplerCache{-1, -1};

      int image(const Material& material) {
        const QString path  = material.albedo.isEmpty() ? QString("$white$") : material.albedo;
        const QString key   = (path.startsWith('$') ? path : canonicalPath(path)) + (material.flipAlbedo ? "|flip" : "|plain");
        const auto existing = imageCache.constFind(key);
        if (existing != imageCache.cend()) {
          return *existing;
        }

        QImage pixels = readTextureImage(path);
        if (pixels.isNull()) {
          fail(QString("Cannot decode texture for GLB: %1").arg(path));
        }

        if (material.flipAlbedo) {
          pixels = pixels.flipped(Qt::Vertical);
        }

        QByteArray png;
        QBuffer output(&png);
        if (!output.open(QIODevice::WriteOnly) || !pixels.save(&output, "PNG")) {
          fail(QString("Cannot encode embedded PNG: %1").arg(path));
        }

        const int index = int(document.images.size());
        document.images.append(QJsonObject{{"name", QFileInfo(path).fileName()}, {"bufferView", document.bytes(png)}, {"mimeType", "image/png"}});
        imageCache.insert(key, index);
        return index;
      }

      int texture(const Material& material) {
        const int source    = image(material);
        const int wrap      = material.clamp ? 1 : 0;
        const quint64 key   = quint64(source) * 2 + wrap;
        const auto existing = textureCache.constFind(key);
        if (existing != textureCache.cend()) {
          return *existing;
        }

        if (samplerCache[wrap] < 0) {
          samplerCache[wrap] = int(document.samplers.size());
          document.samplers.append(QJsonObject{{"magFilter", 9729}, {"minFilter", 9987}, {"wrapS", material.clamp ? 33071 : 10497}, {"wrapT", material.clamp ? 33071 : 10497}});
        }

        const int index = int(document.textures.size());
        document.textures.append(QJsonObject{{"source", source}, {"sampler", samplerCache[wrap]}});
        textureCache.insert(key, index);
        return index;
      }

    public:
      explicit MaterialWriter(glb::Document& value) : document(value) {}

      int add(const Material& material) {
        QJsonArray tint;
        for (int i = 0; i < 3; ++i) {
          const float value = material.tint[i];
          if (!std::isfinite(value) || value < 0 || value > 1) {
            fail(QString("Material '%1' has a tint outside glTF's [0, 1] range").arg(material.name));
          }

          tint.append(value);
        }

        tint.append(1);
        if (material.blend != Blend::None || material.alphaCutoff >= 0 || material.gain != 1 || material.useLightmap || material.useColors || !material.depthTest ||
            !material.depthWrite || material.depthOffset || material.sort != 0) {
          warn(document, "DS2 blend, alpha-test, gain, lightmap, vertex-color and render-order/depth "
                         "features are not represented; materials use the viewer's opaque model appearance.");
        }

        const QJsonObject pbr{{"baseColorFactor", tint}, {"baseColorTexture", QJsonObject{{"index", texture(material)}}}, {"metallicFactor", 0}, {"roughnessFactor", 1}};
        const int index = int(document.materials.size());
        document.materials.append(QJsonObject{{"name", material.name}, {"pbrMetallicRoughness", pbr}, {"alphaMode", "OPAQUE"}, {"doubleSided", !material.cull}});
        return index;
      }
    };

    struct GeometryAccessors {
      QJsonObject attributes;
      QJsonObject skinAttributes;
    };

    struct VertexWeights {
      std::array<quint16, 9> joints{};
      std::array<double, 9> weights{};
      int count    = 0;
      bool zeroSum = false;
    };

    GeometryAccessors writeGeometry(glb::Document& document, const Scene& scene, const Geometry& geometry, const glb::Rig& rig) {
      const size_t vertexCount = size_t(geometry.vertices.size());
      if (!vertexCount || vertexCount > maxGlbSize / (3 * sizeof(float))) {
        fail("Invalid or oversized GLB geometry");
      }

      bool weighted = false;
      for (const auto& vertex : geometry.vertices) {
        weighted |= vertex.influenceCount != 0;
      }

      if (weighted && (rig.skin < 0 || rig.identityJoint < 0)) {
        fail("Weighted geometry has no exportable animation rig");
      }

      if (weighted && rig.identityJoint > std::numeric_limits<quint16>::max()) {
        fail("GLB skin palette, including the identity joint, exceeds 65,536 joints");
      }

      std::vector<VertexWeights> weights;
      int sets = 0;
      if (weighted) {
        weights.resize(vertexCount);
        for (size_t vertexIndex = 0; vertexIndex < vertexCount; ++vertexIndex) {
          const auto& vertex = geometry.vertices[qsizetype(vertexIndex)];
          auto& combined     = weights[vertexIndex];
          if (vertex.influenceCount > 9 || quint64(vertex.influenceStart) + vertex.influenceCount > quint64(scene.influences.size())) {
            fail("Invalid vertex influence range in GLB geometry");
          }

          double sum = 0;
          for (quint32 i = 0; i < vertex.influenceCount; ++i) {
            const auto& influence = scene.influences[qsizetype(quint64(vertex.influenceStart) + i)];
            const double joint = influence.x(), weight = influence.y();
            if (!std::isfinite(joint) || joint < 0 || joint >= rig.identityJoint || std::floor(joint) != joint || !std::isfinite(weight) || weight < 0) {
              fail("Invalid joint index or weight in GLB geometry");
            }

            if (weight == 0) {
              continue;
            }

            int slot = 0;
            while (slot < combined.count && combined.joints[slot] != quint16(joint)) {
              ++slot;
            }

            if (slot == combined.count) {
              combined.joints[combined.count++] = quint16(joint);
            }

            combined.weights[slot] += weight;
            sum += weight;
          }

          if (sum > 0) {
            if (std::abs(sum - 1) > 1e-5) {
              warn(document, "Raw DS2 skin weights were normalized to sum to 1 for glTF; "
                             "the source viewer uses their unnormalized sums, so affected poses may differ.");
            }

            for (int i = 0; i < combined.count; ++i) {
              combined.weights[i] /= sum;
            }
          } else {
            combined.zeroSum    = vertex.influenceCount != 0;
            combined.count      = 1;
            combined.joints[0]  = quint16(rig.identityJoint);
            combined.weights[0] = 1;
            if (combined.zeroSum) {
              warn(document, "Vertices with only zero skin weights were exported at the origin "
                             "with the identity joint, preserving the source shader's zero position.");
            }
          }

          sets = std::max(sets, (combined.count + 3) / 4);
        }
      }

      std::vector<float> positions(vertexCount * 3), normals(vertexCount * 3), uvs(vertexCount * 2);
      for (size_t i = 0; i < vertexCount; ++i) {
        const auto& vertex = geometry.vertices[qsizetype(i)];
        for (int axis = 0; axis < 3; ++axis) {
          if (!std::isfinite(vertex.position[axis])) {
            fail("Cannot export non-finite vertex positions");
          }

          positions[i * 3 + axis] = weighted && weights[i].zeroSum ? 0 : vertex.position[axis];
        }

        double length = 0;
        for (float component : vertex.normal) {
          length += double(component) * component;
        }

        const bool invalidNormal = !std::isfinite(length) || length <= 1e-30;
        if (invalidNormal) {
          warn(document, "Invalid or zero vertex normals were replaced by the source viewer's +Z fallback.");
        }

        const bool fallbackNormal  = invalidNormal || (weighted && weights[i].zeroSum);
        const double inverseLength = fallbackNormal ? 0 : 1 / std::sqrt(length);
        for (int axis = 0; axis < 3; ++axis) {
          normals[i * 3 + axis] = fallbackNormal ? float(axis == 2) : float(vertex.normal[axis] * inverseLength);
        }

        for (int axis = 0; axis < 2; ++axis) {
          if (!std::isfinite(vertex.uv[axis])) {
            fail("Cannot export non-finite texture coordinates");
          }

          uvs[i * 2 + axis] = vertex.uv[axis];
        }
      }

      GeometryAccessors result;
      result.attributes.insert("POSITION", document.floats(positions, "VEC3", true, arrayBuffer));
      result.attributes.insert("NORMAL", document.floats(normals, "VEC3", false, arrayBuffer));
      result.attributes.insert("TEXCOORD_0", document.floats(uvs, "VEC2", false, arrayBuffer));
      if (weighted) {
        std::vector<quint16> joints(vertexCount * 4);
        std::vector<float> values(vertexCount * 4);
        for (int set = 0; set < sets; ++set) {
          std::fill(joints.begin(), joints.end(), quint16(0));
          std::fill(values.begin(), values.end(), 0.f);
          for (size_t vertexIndex = 0; vertexIndex < vertexCount; ++vertexIndex) {
            const auto& vertex = weights[vertexIndex];
            for (int component = 0; component < 4 && set * 4 + component < vertex.count; ++component) {
              joints[vertexIndex * 4 + component] = vertex.joints[set * 4 + component];
              values[vertexIndex * 4 + component] = float(vertex.weights[set * 4 + component]);
            }
          }

          result.skinAttributes.insert(QString("JOINTS_%1").arg(set), document.unsignedShorts(joints, "VEC4", arrayBuffer));
          result.skinAttributes.insert(QString("WEIGHTS_%1").arg(set), document.floats(values, "VEC4", false, arrayBuffer));
        }
      }

      return result;
    }

    void writeAll(QSaveFile& file, const char* data, qint64 size) {
      while (size > 0) {
        const qint64 written = file.write(data, size);
        if (written <= 0) {
          fail(QString("Cannot write GLB: %1").arg(file.errorString()));
        }

        data += written;
        size -= written;
      }
    }

    void save(glb::Document& document, const QJsonObject& json, const QString& path) {
      QByteArray jsonBytes          = QJsonDocument(json).toJson(QJsonDocument::Compact);
      const qsizetype jsonPadding   = (4 - (jsonBytes.size() & 3)) & 3;
      const qsizetype binaryPadding = (4 - (document.binary.size() & 3)) & 3;
      const quint64 jsonLength      = quint64(jsonBytes.size()) + jsonPadding;
      const quint64 binaryLength    = quint64(document.binary.size()) + binaryPadding;
      const quint64 total           = 12 + 8 + jsonLength + 8 + binaryLength;
      if (total > maxGlbSize) {
        fail("GLB exceeds the format's 4 GiB file-size limit");
      }

      jsonBytes.append(jsonPadding, ' ');
      std::array<char, 12> header;
      qToLittleEndian<quint32>(0x46546c67, header.data());
      qToLittleEndian<quint32>(2, header.data() + 4);
      qToLittleEndian<quint32>(quint32(total), header.data() + 8);
      std::array<char, 8> chunk;
      QSaveFile file(path);
      file.setDirectWriteFallback(false);
      if (!file.open(QIODevice::WriteOnly)) {
        fail(QString("Cannot open GLB destination: %1").arg(file.errorString()));
      }

      writeAll(file, header.data(), header.size());
      qToLittleEndian<quint32>(quint32(jsonLength), chunk.data());
      qToLittleEndian<quint32>(0x4e4f534a, chunk.data() + 4);
      writeAll(file, chunk.data(), chunk.size());
      writeAll(file, jsonBytes.constData(), jsonBytes.size());
      qToLittleEndian<quint32>(quint32(binaryLength), chunk.data());
      qToLittleEndian<quint32>(0x004e4942, chunk.data() + 4);
      writeAll(file, chunk.data(), chunk.size());
      writeAll(file, document.binary.constData(), document.binary.size());
      constexpr char zeros[3]{};
      writeAll(file, zeros, binaryPadding);
      if (!file.commit()) {
        fail(QString("Cannot commit GLB destination: %1").arg(file.errorString()));
      }
    }
  } // namespace

  namespace glb {
    int Document::floats(std::span<const float> values, const char* type, bool bounds, int target) {
      const int width = components(type);
      if (values.size() % size_t(width) || values.size() > maxGlbSize / sizeof(float)) {
        fail("Invalid GLB floating-point accessor size");
      }

      const quint64 length = quint64(values.size()) * sizeof(float);
      const auto offset    = reserveBinary(*this, length);
      std::array<float, 16> low, high;
      low.fill(std::numeric_limits<float>::infinity());
      high.fill(-std::numeric_limits<float>::infinity());
      char* destination = binary.data() + offset;
      for (size_t i = 0; i < values.size(); ++i) {
        const float value = values[i];
        if (!std::isfinite(value)) {
          fail("Cannot export non-finite floating-point accessor values");
        }

        quint32 bits;
        std::memcpy(&bits, &value, sizeof(bits));
        qToLittleEndian<quint32>(bits, destination + i * sizeof(float));
        if (bounds) {
          low[i % width]  = std::min(low[i % width], value);
          high[i % width] = std::max(high[i % width], value);
        }
      }

      const int index = accessor(*this, bufferView(*this, offset, length, target), 5126, values.size() / width, type);
      if (bounds) {
        QJsonObject item = accessors[index].toObject();
        QJsonArray minimum, maximum;
        for (int i = 0; i < width; ++i) {
          minimum.append(low[i]);
          maximum.append(high[i]);
        }

        item.insert("min", minimum);
        item.insert("max", maximum);
        accessors[index] = item;
      }

      return index;
    }

    int Document::unsignedShorts(std::span<const quint16> values, const char* type, int target) {
      return integerAccessor(*this, values, type, 5123, target);
    }

    int Document::unsignedInts(std::span<const quint32> values, const char* type, int target) {
      return integerAccessor(*this, values, type, 5125, target);
    }

    int Document::bytes(const QByteArray& values, int target) {
      const auto offset = reserveBinary(*this, quint64(values.size()));
      std::memcpy(binary.data() + offset, values.constData(), size_t(values.size()));
      return bufferView(*this, offset, quint64(values.size()), target);
    }

    void Document::child(int parent, int node) {
      if (parent < 0 || parent >= nodes.size() || node < 0 || node >= nodes.size()) {
        fail("Invalid GLB node hierarchy");
      }

      QJsonObject item    = nodes[parent].toObject();
      QJsonArray children = item.value("children").toArray();
      children.append(node);
      item.insert("children", children);
      nodes[parent] = item;
    }
  } // namespace glb

  QStringList exportGlb(const Scene& scene, const QString& path) {
    if (scene.map) {
      fail("GLB export supports models only, not maps");
    }

    if (path.isEmpty()) {
      fail("GLB destination path is empty");
    }

    if (!scene.path.isEmpty() && canonicalPath(path) == canonicalPath(scene.path)) {
      fail("GLB destination must not overwrite the source model");
    }

    glb::Document document;
    document.warnings  = scene.warnings;
    const glb::Rig rig = glb::writeAnimation(document, scene);
    MaterialWriter materials(document);
    QHash<int, GeometryAccessors> geometryCache;
    for (const auto& mesh : scene.meshes) {
      if (mesh.indices.isEmpty()) {
        warn(document, QString("Empty mesh '%1' was skipped.").arg(mesh.name));
        continue;
      }

      if (mesh.geometry < 0 || mesh.geometry >= scene.geometries.size()) {
        fail(QString("Mesh '%1' refers to missing geometry").arg(mesh.name));
      }

      const auto& geometry = scene.geometries[mesh.geometry];
      if (geometry.vertices.isEmpty()) {
        warn(document, QString("Empty mesh '%1' was skipped.").arg(mesh.name));
        continue;
      }

      if (mesh.indices.size() % 3) {
        fail(QString("Mesh '%1' has an incomplete triangle list").arg(mesh.name));
      }

      bool weighted = false;
      for (quint32 index : mesh.indices) {
        if (quint64(index) >= quint64(geometry.vertices.size())) {
          fail(QString("Mesh '%1' has an invalid vertex index").arg(mesh.name));
        }

        weighted |= geometry.vertices[qsizetype(index)].influenceCount != 0;
      }

      auto cached = geometryCache.constFind(mesh.geometry);
      if (cached == geometryCache.cend()) {
        geometryCache.insert(mesh.geometry, writeGeometry(document, scene, geometry, rig));
        cached = geometryCache.constFind(mesh.geometry);
      }

      QJsonObject attributes = cached->attributes;
      if (weighted) {
        for (auto it = cached->skinAttributes.constBegin(); it != cached->skinAttributes.constEnd(); ++it) {
          attributes.insert(it.key(), it.value());
        }
      }

      const int indices = document.unsignedInts(std::span<const quint32>(mesh.indices.constData(), size_t(mesh.indices.size())), "SCALAR", elementArrayBuffer);
      const QJsonObject primitive{{"attributes", attributes}, {"indices", indices}, {"material", materials.add(mesh.material)}, {"mode", 4}};
      const int meshIndex = int(document.meshes.size());
      document.meshes.append(QJsonObject{{"name", mesh.name}, {"primitives", QJsonArray{primitive}}});
      QJsonObject node{{"name", mesh.name}, {"mesh", meshIndex}};
      if (!mesh.transform.isIdentity()) {
        node.insert("matrix", matrix(mesh.transform));
      }

      if (weighted) {
        node.insert("skin", rig.skin);
      }

      const int nodeIndex = int(document.nodes.size());
      document.nodes.append(node);
      document.roots.append(nodeIndex);
    }

    if (document.meshes.isEmpty()) {
      fail("The model contains no nonempty meshes to export");
    }

    const double halfRoot    = std::sqrt(0.5);
    const int conversionRoot = int(document.nodes.size());
    document.nodes.append(QJsonObject{{"name", scene.name}, {"rotation", QJsonArray{-halfRoot, 0, 0, halfRoot}}, {"children", document.roots}});
    const QJsonObject extras{{"sourceName", scene.name},
                             {"sourceVersion", scene.version},
                             {"sourceFile", QFileInfo(scene.path).fileName()},
                             {"units", "Source units preserved; physical scale is unknown."},
                             {"coordinates", "DS2 Z-up converted to glTF Y-up: [x, z, -y]."},
                             {"warnings", QJsonArray::fromStringList(document.warnings)}};
    QJsonObject json{{"asset", QJsonObject{{"version", "2.0"}, {"generator", "ds2md_view"}, {"extras", extras}}},
                     {"scene", 0},
                     {"scenes", QJsonArray{QJsonObject{{"name", scene.name}, {"nodes", QJsonArray{conversionRoot}}, {"extras", extras}}}},
                     {"nodes", document.nodes},
                     {"meshes", document.meshes},
                     {"materials", document.materials},
                     {"buffers", QJsonArray{QJsonObject{{"byteLength", double(document.binary.size())}}}},
                     {"bufferViews", document.bufferViews},
                     {"accessors", document.accessors}};
    if (!document.images.isEmpty()) {
      json.insert("images", document.images);
      json.insert("textures", document.textures);
      json.insert("samplers", document.samplers);
    }

    if (!document.skins.isEmpty()) {
      json.insert("skins", document.skins);
    }

    if (!document.animations.isEmpty()) {
      json.insert("animations", document.animations);
    }

    save(document, json, path);
    return document.warnings;
  }
} // namespace ds2
