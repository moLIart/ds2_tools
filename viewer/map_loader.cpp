#include "material.h"
#include "scene.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <QDir>
#include <QDirIterator>
#include <QFile>
#include <QFileInfo>
#include <QHash>
#include <QSet>
#include <QtEndian>
#include <stdexcept>

namespace ds2 {
  namespace {
    constexpr uint32_t absent = 0xffffffffu;

    [[noreturn]] void fail(const QString& message) {
      throw std::runtime_error(message.toUtf8().constData());
    }

    QString decode1251(const char* data, qsizetype size) {
      // QString's built-in UTF codecs do not include Windows-1251.
      static constexpr char16_t upper[] = {0x0402, 0x0403, 0x201a, 0x0453, 0x201e, 0x2026, 0x2020, 0x2021, 0x20ac, 0x2030, 0x0409, 0x2039, 0x040a, 0x040c, 0x040b, 0x040f,
                                           0x0452, 0x2018, 0x2019, 0x201c, 0x201d, 0x2022, 0x2013, 0x2014, 0xfffd, 0x2122, 0x0459, 0x203a, 0x045a, 0x045c, 0x045b, 0x045f,
                                           0x00a0, 0x040e, 0x045e, 0x0408, 0x00a4, 0x0490, 0x00a6, 0x00a7, 0x0401, 0x00a9, 0x0404, 0x00ab, 0x00ac, 0x00ad, 0x00ae, 0x0407,
                                           0x00b0, 0x00b1, 0x0406, 0x0456, 0x0491, 0x00b5, 0x00b6, 0x00b7, 0x0451, 0x2116, 0x0454, 0x00bb, 0x0458, 0x0405, 0x0455, 0x0457};
      QString text;
      text.resize(size);
      for (qsizetype i = 0; i < size; ++i) {
        const auto c = static_cast<unsigned char>(data[i]);
        text[i]      = QChar(c < 128 ? char16_t(c) : c < 192 ? upper[c - 128] : char16_t(0x0410 + c - 192));
      }

      return text;
    }

    class Reader {
    public:
      QByteArray data;
      quint64 pos = 0;

      explicit Reader(const QString& path) {
        QFile file(path);
        if (!file.open(QIODevice::ReadOnly)) {
          fail(QString("Cannot open %1: %2").arg(path, file.errorString()));
        }

        data = file.readAll();
        if (file.error() != QFileDevice::NoError) {
          fail(QString("Cannot read %1: %2").arg(path, file.errorString()));
        }
      }

      quint64 take(quint64 size) {
        if (size > quint64(data.size()) - pos) {
          fail(QString("EOF at 0x%1: need %2 bytes, file size %3").arg(pos, 0, 16).arg(size).arg(data.size()));
        }

        const quint64 start = pos;
        pos += size;
        return start;
      }

      uint8_t byte() {
        return uint8_t(data[qsizetype(take(1))]);
      }

      uint16_t word() {
        return qFromLittleEndian<uint16_t>(data.constData() + take(2));
      }

      uint32_t integer() {
        return atU32(take(4));
      }

      float real() {
        return atFloat(take(4));
      }

      uint32_t atU32(quint64 offset) const {
        return qFromLittleEndian<uint32_t>(data.constData() + offset);
      }

      float atFloat(quint64 offset) const {
        const uint32_t bits = atU32(offset);
        float value;
        std::memcpy(&value, &bits, sizeof value);
        return value;
      }

      QVector3D vector() {
        const float x = real(), y = real(), z = real();
        return {x, y, z};
      }

      Bounds bounds() {
        const QVector3D low = vector(), high = vector();
        return {low, high};
      }

      QString string() {
        const auto length = word();
        return decode1251(data.constData() + take(length), length);
      }

      void expect(const QString& expected) {
        const auto offset = pos;
        const auto actual = string();
        if (actual != expected) {
          fail(QString("At 0x%1: expected '%2', got '%3'").arg(offset, 0, 16).arg(expected, actual));
        }
      }

      uint32_t count(quint64 minimum) {
        const uint32_t n = integer();
        if (n > (quint64(data.size()) - pos) / minimum || n > uint32_t(std::numeric_limits<int>::max())) {
          fail(QString("Impossible count %1 at 0x%2").arg(n).arg(pos - 4, 0, 16));
        }

        return n;
      }
    };

    struct Buffer {
      quint64 offset = 0;
      uint32_t flags = 0, count = 0;
      quint64 indices = 0, positions = 0, uv = 0, lightUv = 0, colors = 0;
    };

    struct DrawRange {
      uint32_t count = 0, buffer = absent, start = 0;
    };

    struct MapMesh {
      quint64 offset = 0;
      bool instanced = false;
      QString material;
      std::array<QString, 5> textures;
      Bounds bounds;
      uint32_t lightmap = absent, vertexCount = 0, vertexBuffer = absent, vertexStart = 0;
      uint32_t colorBuffer = absent, colorStart = 0;
      uint32_t modelGroup = absent, modelInstance = absent;
      DrawRange list, strip;
      QMatrix4x4 matrix;
    };

    struct ParsedMap {
      QVector<MapMesh> meshes;
      QVector<Buffer> buffers;
      QStringList lightmaps;
      QVector<VisNode> nodes;
      QVector<Light> lights;
      QVector<MapModel> models;
    };

    bool finiteBounds(const Bounds& bounds) {
      for (int i = 0; i < 3; ++i) {
        if (!std::isfinite(bounds.low[i]) || !std::isfinite(bounds.high[i])) {
          return false;
        }
      }

      return true;
    }

    ParsedMap parse(Reader& r) {
      r.expect("DS2GeometryFile_1");
      const auto version = r.string();
      if (version != "0.8") {
        fail(QString("Unsupported DS2 geometry version '%1'").arg(version));
      }

      ParsedMap out;
      QSet<QString> seen;
      while (r.pos < quint64(r.data.size())) {
        const auto chunkOffset = r.pos;
        r.expect("DS2Chunk_begin");
        const QString name = r.string();
        if (seen.contains(name)) {
          fail(QString("Duplicate chunk '%1' at 0x%2").arg(name).arg(chunkOffset, 0, 16));
        }

        seen.insert(name);
        if (name == "meshes") {
          const auto count = r.count(270);
          out.meshes.reserve(count);
          for (uint32_t i = 0; i < count; ++i) {
            MapMesh mesh;
            mesh.offset    = r.pos;
            mesh.instanced = r.byte() != 0;
            mesh.material  = r.string();
            for (auto& texture : mesh.textures) {
              texture = r.string();
            }

            mesh.bounds = r.bounds();
            r.take(16 + 156 + 1 + 12); // Sphere, auxiliary bounds, flag_114, vector_118.
            mesh.lightmap = r.integer();
            r.integer(); // primitive_count is advisory; both stored index streams render.
            mesh.vertexCount = r.integer();
            r.integer(); // field_d8
            mesh.list         = {r.integer(), r.integer(), r.integer()};
            mesh.strip        = {r.integer(), r.integer(), r.integer()};
            mesh.vertexBuffer = r.integer();
            mesh.vertexStart  = r.integer();
            if (mesh.instanced) {
              mesh.colorBuffer = r.integer();
              mesh.colorStart  = r.integer();
              float matrix[16];
              for (float& value : matrix) {
                value = r.real();
                if (!std::isfinite(value) || std::abs(value) > 1e9f) {
                  fail(QString("Unsafe instance matrix in mesh %1 at 0x%2").arg(i).arg(mesh.offset, 0, 16));
                }
              }

              mesh.matrix        = QMatrix4x4(matrix); // File rows; Qt constructor converts to column-major storage.
              mesh.modelGroup    = r.integer();
              mesh.modelInstance = r.integer();
            }

            out.meshes.append(std::move(mesh));
          }
        } else if (name == "buffers") {
          const auto count = r.count(8);
          out.buffers.reserve(count);
          for (uint32_t i = 0; i < count; ++i) {
            Buffer buffer;
            buffer.offset = r.pos;
            buffer.flags  = r.integer();
            if (!buffer.flags || (buffer.flags & ~255u)) {
              fail(QString("Unknown buffer flags 0x%1 at 0x%2").arg(buffer.flags, 0, 16).arg(buffer.offset, 0, 16));
            }

            buffer.count    = r.integer();
            const quint64 n = buffer.count;
            if (buffer.flags & 1) {
              buffer.indices = r.take(n * 4);
            } else {
              if (buffer.flags & 2) {
                buffer.positions = r.take(n * 12);
              }

              if (buffer.flags & 8) {
                buffer.uv = r.take(n * 8);
              }

              if (buffer.flags & 16) {
                buffer.lightUv = r.take(n * 8);
              }

              if (buffer.flags & 32) {
                buffer.colors = r.take(n * 16);
              }

              if (buffer.flags & 4) {
                r.take(n * 12);
              }

              if (buffer.flags & 64) {
                r.take(n * 12);
              }

              if (buffer.flags & 128) {
                r.take(n * 12);
              }
            }

            out.buffers.append(buffer);
          }
        } else if (name == "lightmaps") {
          const auto count = r.count(2);
          out.lightmaps.reserve(count);
          for (uint32_t i = 0; i < count; ++i) {
            out.lightmaps.append(r.string());
          }
        } else if (name == "vistree") {
          const auto count = r.count(45);
          out.nodes.reserve(count);
          for (uint32_t i = 0; i < count; ++i) {
            VisNode node;
            node.bounds = r.bounds();
            r.take(16); // Radius and center.
            const bool leaf = r.byte() != 0;
            if (leaf) {
              const auto meshCount = r.count(4);
              node.meshes.reserve(meshCount);
              for (uint32_t j = 0; j < meshCount; ++j) {
                const auto id = r.integer();
                if (id > uint32_t(std::numeric_limits<int>::max())) {
                  fail(QString("Vistree mesh %1 outside mesh table").arg(id));
                }

                node.meshes.append(int(id));
              }
            } else {
              node.children.reserve(8);
              for (int j = 0; j < 8; ++j) {
                const auto id = r.integer();
                if (id == absent) {
                  continue;
                }

                if (id > uint32_t(std::numeric_limits<int>::max())) {
                  fail(QString("Vistree child %1 outside node table").arg(id));
                }

                node.children.append(int(id));
              }
            }

            out.nodes.append(std::move(node));
          }
        } else if (name == "lights") {
          const auto count = r.count(72);
          out.lights.reserve(count);
          for (uint32_t i = 0; i < count; ++i) {
            const auto type = r.integer();
            if (type < uint32_t(LightType::Point) || type > uint32_t(LightType::Spot)) {
              fail(QString("Unknown light type %1 at 0x%2").arg(type).arg(r.pos - 4, 0, 16));
            }

            Light light;
            light.type      = static_cast<LightType>(type);
            light.position  = r.vector();
            light.direction = r.vector();
            light.color     = r.vector();
            light.range     = r.real();
            r.take(28); // Angles and remaining parameters are not needed by the gizmos.
            out.lights.append(light);
          }
        } else if (name == "models") {
          const auto count = r.count(30);
          out.models.reserve(count);
          for (uint32_t i = 0; i < count; ++i) {
            MapModel model;
            model.name           = r.string();
            model.bounds         = r.bounds();
            const auto instances = r.count(88);
            model.instances.reserve(instances);
            for (uint32_t j = 0; j < instances; ++j) {
              float matrix[16];
              for (float& value : matrix) {
                value = r.real();
              }

              model.instances.append({QMatrix4x4(matrix), r.bounds()});
            }

            out.models.append(std::move(model));
          }
        } else {
          fail(QString("Unknown chunk '%1' at 0x%2; no size to safely skip it").arg(name).arg(chunkOffset, 0, 16));
        }

        r.expect("DS2Chunk_end");
      }

      if (!seen.contains("meshes") || !seen.contains("buffers") || !seen.contains("lightmaps")) {
        fail("Missing geometry, buffers or lightmaps chunk");
      }

      return out;
    }

    void bindModelMeshes(Scene& scene, const QVector<MapMesh>& sources) {
      int unresolved = 0;
      for (qsizetype meshId = 0; meshId < sources.size(); ++meshId) {
        const auto& source = sources[meshId];
        if (!source.instanced) {
          continue;
        }

        const float* matrix = source.matrix.constData();
        auto matchesMatrix = [&](const MapModelInstance& instance) {
          return std::equal(matrix, matrix + 16, instance.transform.constData());
        };
        int groupId = -1, instanceId = -1;
        auto findMatches = [&](int firstGroup, int lastGroup) {
          int matches = 0;
          for (int group = firstGroup; group < lastGroup; ++group) {
            const auto& instances = scene.models[group].instances;
            for (qsizetype instance = 0; instance < instances.size(); ++instance) {
              if (matchesMatrix(instances[instance])) {
                groupId    = group;
                instanceId = int(instance);
                if (++matches == 2) {
                  return matches; // Ambiguous transforms must not select unrelated geometry.
                }
              }
            }
          }

          return matches;
        };
        int matches = 0;
        if (source.modelGroup < uint32_t(scene.models.size())) {
          const auto& group = scene.models[int(source.modelGroup)];
          if (source.modelInstance < uint32_t(group.instances.size()) && matchesMatrix(group.instances[int(source.modelInstance)])) {
            groupId    = int(source.modelGroup);
            instanceId = int(source.modelInstance);
            matches    = 1;
          } else {
            matches = findMatches(int(source.modelGroup), int(source.modelGroup) + 1);
          }
        }

        if (!matches) {
          matches = findMatches(0, int(scene.models.size()));
        }

        if (matches == 1) {
          scene.models[groupId].instances[instanceId].meshes.append(int(meshId));
        } else {
          ++unresolved;
        }
      }

      if (unresolved) {
        scene.warnings.append(QString("Model hierarchy: %1 meshes have no unique matching instance; omitted from selection").arg(unresolved));
      }
    }

    using Binding = std::array<uint32_t, 4>;

    struct BindingInfo {
      uint32_t count = 0;
      int geometry   = -1;
    };

    std::map<Binding, BindingInfo> validate(const ParsedMap& map, const Reader& r) {
      for (const auto& buffer : map.buffers) {
        for (const auto& attribute : {std::pair{buffer.positions, 3u}, std::pair{buffer.uv, 2u}, std::pair{buffer.lightUv, 2u}}) {
          if (!attribute.first) {
            continue;
          }

          const QString name  = attribute.first == buffer.positions ? "positions" : attribute.first == buffer.uv ? "uv" : "lightmap_uv";
          const quint64 count = quint64(buffer.count) * attribute.second;
          for (quint64 i = 0; i < count; ++i) {
            const float value = r.atFloat(attribute.first + i * 4);
            if (!std::isfinite(value) || (attribute.second == 3 && std::abs(value) > 1e9f)) {
              fail(QString("Unsafe %1 coordinates in buffer at 0x%2").arg(name).arg(buffer.offset, 0, 16));
            }
          }
        }
      }

      std::map<Binding, BindingInfo> bindings;
      for (qsizetype number = 0; number < map.meshes.size(); ++number) {
        const auto& mesh    = map.meshes[number];
        const auto meshFail = [&](const QString& message) {
          fail(QString("Mesh %1 at 0x%2: %3").arg(number).arg(mesh.offset, 0, 16).arg(message));
        };
        if (!finiteBounds(mesh.bounds)) {
          meshFail("non-finite bounds");
        }

        if (mesh.vertexBuffer >= map.buffers.size()) {
          meshFail("vertex_buffer outside buffers");
        }

        const auto& vb = map.buffers[mesh.vertexBuffer];
        if (!vb.positions) {
          meshFail("vertex buffer has no positions");
        }

        if (quint64(mesh.vertexStart) + mesh.vertexCount > vb.count) {
          meshFail("vertex range outside buffer");
        }

        if (mesh.lightmap != absent && mesh.lightmap >= map.lightmaps.size()) {
          meshFail("lightmap index outside table");
        }

        uint32_t required = mesh.vertexCount;
        for (const auto& range : {mesh.list, mesh.strip}) {
          if (!range.count) {
            continue;
          }

          if (range.buffer >= map.buffers.size() || !map.buffers[range.buffer].indices) {
            meshFail("index_buffer outside index buffers");
          }

          const auto& ib = map.buffers[range.buffer];
          if (quint64(range.start) + range.count > ib.count) {
            meshFail("index range outside buffer");
          }

          for (quint64 i = 0; i < range.count; ++i) {
            const auto index = r.atU32(ib.indices + (quint64(range.start) + i) * 4);
            if (quint64(index) + mesh.vertexStart >= vb.count) {
              meshFail("index plus vertex base outside vertex buffer");
            }

            required = std::max(required, index + 1);
          }
        }

        const uint32_t colorBuffer = mesh.colorBuffer == absent ? mesh.vertexBuffer : mesh.colorBuffer;
        const uint32_t colorStart  = mesh.colorBuffer == absent ? mesh.vertexStart : mesh.colorStart;
        if (colorBuffer >= map.buffers.size()) {
          meshFail("instance color buffer outside buffer table");
        }

        const auto& colors = map.buffers[colorBuffer];
        if (mesh.colorBuffer != absent && !colors.colors) {
          meshFail("instance color buffer has no colors");
        }

        if (colors.colors && quint64(colorStart) + required > colors.count) {
          meshFail("instance color range outside color buffer");
        }

        auto& binding = bindings[{mesh.vertexBuffer, mesh.vertexStart, colorBuffer, colorStart}];
        binding.count = std::max(binding.count, required);
      }

      if (!map.nodes.isEmpty()) {
        QVector<uint8_t> visited(map.nodes.size(), 0), covered(map.meshes.size(), 0);
        QVector<int> pending{0};
        qsizetype visitedCount = 0, coveredCount = 0;
        while (!pending.isEmpty()) {
          const int id = pending.takeLast();
          if (id < 0 || id >= map.nodes.size()) {
            fail(QString("Vistree child %1 outside node table").arg(id));
          }

          if (visited[id]) {
            fail(QString("Vistree node %1 has a cycle or multiple parents").arg(id));
          }

          visited[id] = 1;
          ++visitedCount;
          const auto& node = map.nodes[id];
          bool valid       = finiteBounds(node.bounds);
          for (int axis = 0; axis < 3; ++axis) {
            valid = valid && node.bounds.low[axis] <= node.bounds.high[axis];
          }

          if (!valid) {
            fail(QString("Vistree node %1 has invalid bounds").arg(id));
          }

          for (const auto child : node.children) {
            pending.append(child);
          }

          for (const auto mesh : node.meshes) {
            if (mesh < 0 || mesh >= map.meshes.size()) {
              fail(QString("Vistree mesh %1 outside mesh table").arg(mesh));
            }

            if (!covered[mesh]) {
              covered[mesh] = 1;
              ++coveredCount;
            }
          }
        }

        if (visitedCount != map.nodes.size()) {
          fail("Vistree contains nodes unreachable from root 0");
        }

        if (coveredCount != map.meshes.size()) {
          fail("Vistree does not cover every mesh");
        }
      }

      return bindings;
    }

    class TextureIndex {
      QHash<QString, QStringList> images;

    public:
      explicit TextureIndex(const QString& root) {
        QStringList paths;
        QDirIterator it(root, QDir::Files, QDirIterator::Subdirectories);
        while (it.hasNext()) {
          const QString path   = it.next();
          const auto extension = QFileInfo(path).suffix().toCaseFolded();
          if (extension == "dds" || extension == "tga") {
            paths.append(QFileInfo(path).absoluteFilePath());
          }
        }

        std::sort(paths.begin(), paths.end());
        for (const auto& path : paths) {
          images[QFileInfo(path).fileName().toCaseFolded()].append(path);
        }
      }

      QString resolve(const QString& reference, const QHash<QString, QStringList>& siblings, QStringList& warnings) const {
        QString source = reference;
        source.replace('\\', '/');
        const auto parts = source.split('/', Qt::SkipEmptyParts);
        const auto name  = parts.isEmpty() ? QString() : parts.last();
        if (name.isEmpty() || reference.toCaseFolded() == "notexture") {
          warnings.append(QString("No texture for '%1'").arg(reference));
          return "$missing$";
        }

        const auto stem = QFileInfo(name).completeBaseName();
        const QStringList names{(stem + ".dds").toCaseFolded(), (stem + ".tga").toCaseFolded()};
        for (const auto& candidate : names) {
          const auto found = siblings.constFind(candidate);
          if (found != siblings.cend() && found->size() == 1) {
            return found->first();
          }
        }

        for (const auto& candidate : names) {
          const auto found = images.constFind(candidate);
          if (found == images.cend()) {
            continue;
          }

          if (found->size() == 1) {
            return found->first();
          }

          QString exact;
          int exactCount = 0;
          if (parts.size() >= 2) {
            for (const auto& path : *found) {
              if (QFileInfo(path).dir().dirName().toCaseFolded() == parts[parts.size() - 2].toCaseFolded()) {
                exact = path;
                ++exactCount;
              }
            }
          }

          if (exactCount == 1) {
            return exact;
          }

          warnings.append(QString("Ambiguous local texture '%1': %2 candidates").arg(reference).arg(found->size()));
          return "$missing$";
        }

        warnings.append(QString("Missing local DDS/TGA for '%1'").arg(reference));
        return "$missing$";
      }
    };

    struct MapResources {
      QHash<QString, MaterialDeclaration> materials;
      TextureIndex textures;

      explicit MapResources(const QString& root) : materials(cachedMaterials(QDir(root).filePath("gameres/materials"))), textures(QDir(root).filePath("gameres/textures")) {}
    };

    MapResources& resourcesFor(const QString& gameRoot) {
      // Loading is synchronous on the UI thread. Keep the viewer's original resource
      // snapshot and undefined-material fallbacks when switching maps in one game.
      static std::map<QString, std::unique_ptr<MapResources>> resources;
      const QString root = QDir::cleanPath(QDir(gameRoot).absolutePath());
#ifdef Q_OS_WIN
      const QString key = root.toCaseFolded();
#else
      const QString& key = root;
#endif
      auto found = resources.find(key);
      if (found == resources.end()) {
        found = resources.emplace(key, std::make_unique<MapResources>(root)).first;
      }

      return *found->second;
    }

    void appendIndices(QVector<uint32_t>& indices, const MapMesh& mesh, const ParsedMap& map, const Reader& r) {
      const quint64 triangles = quint64(mesh.list.count / 3) + (mesh.strip.count >= 3 ? quint64(mesh.strip.count) - 2 : 0);
      if (triangles * 3 > quint64(std::numeric_limits<int>::max())) {
        fail("Expanded mesh index count is too large");
      }

      indices.reserve(qsizetype(triangles * 3));
      if (mesh.list.count) {
        const auto offset = map.buffers[mesh.list.buffer].indices + quint64(mesh.list.start) * 4;
        // GL_TRIANGLES ignores a trailing incomplete primitive; do not join it to the strip.
        for (quint64 i = 0; i < quint64(mesh.list.count / 3) * 3; ++i) {
          indices.append(r.atU32(offset + i * 4));
        }
      }

      if (mesh.strip.count >= 3) {
        const auto offset = map.buffers[mesh.strip.buffer].indices + quint64(mesh.strip.start) * 4;
        uint32_t a = r.atU32(offset), b = r.atU32(offset + 4);
        for (quint64 i = 2; i < mesh.strip.count; ++i) {
          const uint32_t c = r.atU32(offset + i * 4);
          // Degenerates advance parity too. Preserve them to retain the stored topology.
          indices.append((i & 1) ? b : a);
          indices.append((i & 1) ? a : b);
          indices.append(c);
          a = b;
          b = c;
        }
      }
    }

    struct Plane {
      double a, b, c, d;
    };

    bool inFrustum(const Bounds& bounds, const std::array<Plane, 6>& planes) {
      for (const auto& p : planes) {
        const double x         = p.a * (p.a >= 0 ? bounds.high.x() : bounds.low.x());
        const double y         = p.b * (p.b >= 0 ? bounds.high.y() : bounds.low.y());
        const double z         = p.c * (p.c >= 0 ? bounds.high.z() : bounds.low.z());
        const double tolerance = 1e-6 * (std::abs(x) + std::abs(y) + std::abs(z) + std::abs(p.d));
        if (x + y + z + p.d < -tolerance) {
          return false;
        }
      }

      return true;
    }
  } // namespace

  Scene loadMap(const QString& path, const QString& gameRoot) {
    Reader reader(path);
    auto map       = parse(reader);
    auto bindings  = validate(map, reader);
    if (map.meshes.isEmpty()) {
      fail("Map contains no geometry");
    }

    Scene scene;
    scene.path           = QFileInfo(path).absoluteFilePath();
    scene.name           = QFileInfo(path).completeBaseName();
    scene.version        = "0.8";
    scene.map            = true;
    scene.lights         = std::move(map.lights);
    scene.models         = std::move(map.models);
    scene.note           = "Static geometry, albedo, lightmaps and instance vertex lighting; game effects and scripted "
                           "objects are not reproduced.";
    qsizetype instanceCount = 0;
    for (const auto& model : scene.models) {
      instanceCount += model.instances.size();
    }

    scene.note += QString("\nModels: %1 groups, %2 instances; M toggles geometry; select groups or instances in the model tree to show AABBs and axes")
                      .arg(scene.models.size()).arg(instanceCount);
    auto& resources      = resourcesFor(gameRoot);
    auto& materials      = resources.materials;
    const auto& textures = resources.textures;
    QHash<QString, QStringList> nearby;
    for (const auto& image : QDir(QFileInfo(path).absolutePath()).entryInfoList(QDir::Files, QDir::Name)) {
      nearby[image.fileName().toCaseFolded()].append(image.absoluteFilePath());
    }

    QHash<QString, QString> albedos;
    QStringList lightmaps;
    // Match Python warning order: mesh materials/albedos first, then lightmaps.
    scene.meshes.reserve(map.meshes.size());
    scene.geometries.reserve(qsizetype(bindings.size()));
    scene.bounds = map.meshes.first().bounds;
    for (qsizetype i = 0; i < map.meshes.size(); ++i) {
      const auto& source      = map.meshes[i];
      const auto materialName = source.material.toCaseFolded();
      auto declaration        = materials.constFind(materialName);
      if (declaration == materials.cend()) {
        MaterialDeclaration fallback;
        fallback.name = source.material;
        scene.warnings.append(QString("Undefined material '%1': showing static texture only").arg(source.material));
        materials.insert(materialName, fallback);
        declaration = materials.constFind(materialName);
      }

      const auto& d           = *declaration;
      const QString reference = d.diffuse == "$diffuse$" ? source.textures[0] : d.diffuse;
      if (!albedos.contains(reference)) {
        albedos.insert(reference, reference == "$white$" || reference == "$black$" ? reference : textures.resolve(reference, nearby, scene.warnings));
      }

      if (d.shader == "envmap" || d.shader == "embm" || d.shader == "space_distortion" || d.shader == "flare" || materialName.contains("refl") ||
          materialName.startsWith("cubeman")) {
        scene.warnings.append(QString("%1: static albedo/lighting only; reflection/distortion/flare effects not reproduced").arg(source.material));
      }

      const uint32_t colorBuffer = source.colorBuffer == absent ? source.vertexBuffer : source.colorBuffer;
      const uint32_t colorStart  = source.colorBuffer == absent ? source.vertexStart : source.colorStart;
      const Binding key{source.vertexBuffer, source.vertexStart, colorBuffer, colorStart};
      auto& binding = bindings.at(key);
      if (binding.geometry < 0) {
        const auto& vb = map.buffers[source.vertexBuffer];
        const auto& cb = map.buffers[colorBuffer];
        Geometry geometry;
        geometry.hasColors = cb.colors != 0;
        geometry.vertices.resize(binding.count);
        for (quint64 vertex = 0; vertex < binding.count; ++vertex) {
          auto& out          = geometry.vertices[qsizetype(vertex)];
          const quint64 base = quint64(source.vertexStart) + vertex;
          for (int component = 0; component < 3; ++component) {
            out.position[component] = reader.atFloat(vb.positions + base * 12 + component * 4);
          }

          for (int component = 0; component < 2; ++component) {
            if (vb.uv) {
              out.uv[component] = reader.atFloat(vb.uv + base * 8 + component * 4);
            }

            if (vb.lightUv) {
              out.lightUv[component] = reader.atFloat(vb.lightUv + base * 8 + component * 4);
            }
          }

          // The map shader never uses exported normals/tangents (unused vectors can be NaN).
          if (cb.colors) {
            for (int component = 0; component < 4; ++component) {
              out.color[component] = float(reader.atU32(cb.colors + (quint64(colorStart) + vertex) * 16 + component * 4) & 255u) / 255.f;
            }
          }
        }

        binding.geometry = int(scene.geometries.size());
        scene.geometries.append(std::move(geometry));
      }

      Mesh mesh;
      mesh.name         = QString("%1 [%2]").arg(source.material).arg(i);
      mesh.geometry     = binding.geometry;
      mesh.bounds       = source.bounds; // Already world space, including the instance matrix.
      mesh.transform    = source.matrix;
      mesh.sortGeometry = int(source.vertexBuffer);
      mesh.mapModel     = source.instanced;
      appendIndices(mesh.indices, source, map, reader);
      auto& m       = mesh.material;
      m.name        = d.name;
      m.albedo      = albedos.value(reference);
      m.flipAlbedo  = QFileInfo(m.albedo).suffix().toCaseFolded() == "tga";
      m.tint        = d.color;
      m.diffuseTransform = d.diffuseTransform;
      m.alphaCutoff = d.alphaCutoff;
      m.sort        = float(d.sort);
      m.cull        = d.cull;
      m.depthTest   = d.depthTest;
      m.depthWrite  = d.depthWrite;
      m.depthOffset = d.depthOffset;
      m.clamp       = d.clamp;
      m.useLightmap = source.lightmap != absent && !d.selfIllumination;
      m.useColors   = !d.selfIllumination && (!m.useLightmap || d.vertexColors);
      m.gain        = d.selfIllumination ? 1.f : m.useLightmap || d.overbright ? 2.f : 1.f;
      if (d.blend == "blend") {
        m.blend = Blend::Alpha;
      } else if (d.blend == "add") {
        m.blend = Blend::Add;
      } else if (d.blend == "modulate") {
        m.blend = Blend::Modulate;
      } else if (!d.blend.isEmpty()) {
        m.blend = Blend::Replace;
      }

      scene.meshes.append(std::move(mesh));
      for (int axis = 0; axis < 3; ++axis) {
        scene.bounds.low[axis]  = std::min(scene.bounds.low[axis], source.bounds.low[axis]);
        scene.bounds.high[axis] = std::max(scene.bounds.high[axis], source.bounds.high[axis]);
      }
    }

    bindModelMeshes(scene, map.meshes);

    lightmaps.reserve(map.lightmaps.size());
    for (const auto& reference : map.lightmaps) {
      lightmaps.append(textures.resolve(reference, nearby, scene.warnings));
    }

    for (qsizetype i = 0; i < map.meshes.size(); ++i) {
      const auto index = map.meshes[i].lightmap;
      if (index != absent) {
        auto& material        = scene.meshes[i].material;
        material.lightmap     = lightmaps[index];
        material.flipLightmap = QFileInfo(material.lightmap).suffix().toCaseFolded() == "tga";
      }
    }

    scene.visibility = map.nodes;
    scene.warnings.removeDuplicates();
    return scene;
  }

  void visibleMeshes(const Scene& scene, const QMatrix4x4& zeroToOneClip, QVector<int>& result) {
    const auto row = [&](int axis, int sign, bool addW) {
      return Plane{
          (addW ? double(zeroToOneClip(3, 0)) : 0.) + sign * double(zeroToOneClip(axis, 0)), (addW ? double(zeroToOneClip(3, 1)) : 0.) + sign * double(zeroToOneClip(axis, 1)),
          (addW ? double(zeroToOneClip(3, 2)) : 0.) + sign * double(zeroToOneClip(axis, 2)), (addW ? double(zeroToOneClip(3, 3)) : 0.) + sign * double(zeroToOneClip(axis, 3))};
    };
    const std::array<Plane, 6> planes{row(0, 1, true), row(0, -1, true), row(1, 1, true), row(1, -1, true), row(2, 1, false), row(2, -1, true)};
    result.clear();
    result.reserve(scene.meshes.size());
    if (scene.visibility.isEmpty()) {
      for (qsizetype i = 0; i < scene.meshes.size(); ++i) {
        if (inFrustum(scene.meshes[i].bounds, planes)) {
          result.append(int(i));
        }
      }

      return;
    }

    // Calls are synchronous on the render thread; retain traversal scratch without shared scene mutation.
    thread_local QVector<int> pending;
    thread_local QVector<uint32_t> stamps;
    thread_local uint32_t generation = 0;
    if (stamps.size() < scene.meshes.size()) {
      stamps.resize(scene.meshes.size());
    }

    if (++generation == 0) {
      std::fill(stamps.begin(), stamps.end(), 0);
      generation = 1;
    }

    for (const int id : scene.movedMeshes) {
      stamps[id] = generation;
      if (inFrustum(scene.meshes[id].bounds, planes)) {
        result.append(id);
      }
    }

    pending.clear();
    pending.append(0);
    while (!pending.isEmpty()) {
      const auto& node = scene.visibility[pending.takeLast()];
      if (!inFrustum(node.bounds, planes)) {
        continue;
      }

      for (const auto child : node.children) {
        pending.append(child);
      }

      for (const auto id : node.meshes) {
        if (stamps[id] == generation) {
          continue;
        }

        stamps[id] = generation;
        // A leaf merely intersects this world AABB. Never transform it or inherit the leaf result.
        if (inFrustum(scene.meshes[id].bounds, planes)) {
          result.append(id);
        }
      }
    }
  }
} // namespace ds2
