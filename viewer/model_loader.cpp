#include "model_animation.h"
#include "scene.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <QDir>
#include <QDirIterator>
#include <QFile>
#include <QFileInfo>
#include <QHash>
#include <QRegularExpression>
#include <QtEndian>
#include <stdexcept>
#include <utility>

namespace ds2 {
  namespace {
    [[noreturn]] void fail(const QString& message) {
      throw std::runtime_error(message.toStdString());
    }

    QString cp1251(const QByteArray& bytes) {
      static constexpr char16_t high[] = u"\u0402\u0403\u201a\u0453\u201e\u2026\u2020\u2021\u20ac\u2030\u0409\u2039\u040a\u040c\u040b\u040f"
                                         u"\u0452\u2018\u2019\u201c\u201d\u2022\u2013\u2014\ufffd\u2122\u0459\u203a\u045a\u045c\u045b\u045f"
                                         u"\u00a0\u040e\u045e\u0408\u00a4\u0490\u00a6\u00a7\u0401\u00a9\u0404\u00ab\u00ac\u00ad\u00ae\u0407"
                                         u"\u00b0\u00b1\u0406\u0456\u0491\u00b5\u00b6\u00b7\u0451\u2116\u0454\u00bb\u0458\u0405\u0455\u0457";
      QString result;
      result.reserve(bytes.size());
      for (unsigned char c : bytes) {
        result.append(QChar(c < 128 ? c : c < 192 ? high[c - 128] : 0x410 + c - 192));
      }

      return result;
    }

    bool space(char c) {
      return c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\v' || c == '\f';
    }

    class Reader {
      const QByteArray& data;

    public:
      qsizetype pos = 0;
      bool text     = false;

      explicit Reader(const QByteArray& bytes) : data(bytes) {}

      qsizetype remaining() const {
        return data.size() - pos;
      }

      const char* take(quint64 size) {
        if (size > quint64(remaining())) {
          fail(QString("EOF at 0x%1: need %2 bytes, file size %3").arg(pos, 0, 16).arg(size).arg(data.size()));
        }

        const char* p = data.constData() + pos;
        pos += qsizetype(size);
        return p;
      }

      void whitespace() {
        while (pos < data.size() && space(data[pos])) {
          ++pos;
        }
      }

      QByteArray token(bool quoted) {
        whitespace();
        const auto start = pos;
        if (!remaining()) {
          fail(QString("EOF at 0x%1: expected text token").arg(pos, 0, 16));
        }

        qsizetype begin = pos;
        if (quoted) {
          if (data[pos++] != '"') {
            fail(QString("Expected quoted string at 0x%1").arg(start, 0, 16));
          }

          begin = pos;
          while (pos < data.size() && data[pos] != '"') {
            ++pos;
          }

          if (!remaining()) {
            fail(QString("Unterminated quoted string at 0x%1").arg(start, 0, 16));
          }
        } else {
          while (pos < data.size() && !space(data[pos])) {
            ++pos;
          }
        }

        const auto end = pos;
        if (quoted) {
          ++pos;
        }

        if (remaining() && !space(data[pos])) {
          fail(QString("Missing token separator at 0x%1").arg(pos, 0, 16));
        }

        return data.mid(begin, end - begin);
      }

      qint64 integer() {
        const auto bytes = token(false);
        static const QRegularExpression pattern("^[+-]?[0-9]+$");
        bool ok            = false;
        const qint64 value = bytes.toLongLong(&ok);
        if (!ok || !pattern.match(QString::fromLatin1(bytes)).hasMatch()) {
          fail("Invalid integer token");
        }

        return value;
      }

      quint32 count() {
        const qint64 n = integer();
        if (n < 0 || quint64(n) > std::numeric_limits<quint32>::max()) {
          fail("Invalid or negative text count");
        }

        return quint32(n);
      }

      quint8 u8() {
        return text ? quint8(countRange(255)) : quint8(*take(1));
      }

      quint16 u16() {
        return text ? quint16(countRange(65535)) : qFromLittleEndian<quint16>(take(2));
      }

      quint32 u32() {
        return text ? count() : qFromLittleEndian<quint32>(take(4));
      }

      quint32 countRange(quint32 maximum) {
        const quint32 n = count();
        if (n > maximum) {
          fail("Text integer is outside supported range");
        }

        return n;
      }

      double real() {
        if (!text) {
          const quint32 bits = qFromLittleEndian<quint32>(take(4));
          float value;
          std::memcpy(&value, &bits, sizeof(value));
          return value;
        }

        const auto bytes = token(false);
        static const QRegularExpression pattern("^[+-]?(?:[0-9]+(?:\\.[0-9]*)?|\\.[0-9]+)(?:[eE][+-]?[0-9]+)?$");
        bool ok            = false;
        const double value = bytes.toDouble(&ok);
        if (!ok || !std::isfinite(value) || !pattern.match(QString::fromLatin1(bytes)).hasMatch()) {
          fail("Invalid or non-finite real token");
        }

        return value;
      }

      QString string() {
        if (text) {
          return cp1251(token(true));
        }

        const auto size = u16();
        return cp1251(QByteArray(take(size), size));
      }

      void ensureValues(quint64 count, quint64 stride) const {
        if (count > quint64(remaining()) / (text ? 1 : stride)) {
          fail("Array extends beyond model data");
        }
      }

      void end() {
        if (text) {
          whitespace();
        }

        if (remaining()) {
          fail(QString("%1 trailing bytes at 0x%2").arg(remaining()).arg(pos, 0, 16));
        }
      }
    };

    void finite(double v, const char* label) {
      if (!std::isfinite(v)) {
        fail("Non-finite " + QString::fromLatin1(label));
      }
    }

    void zero(qint64 value, const QString& label) {
      if (value != 0) {
        fail(QString("Unsupported %1 %2").arg(label).arg(value));
      }
    }

    double rate(double value) {
      if (!std::isfinite(value) || value <= 0) {
        fail("Invalid sampled animation rate");
      }

      return value;
    }

    model::Matrix matrix(Reader& r, bool check = true) {
      r.ensureValues(16, 4);
      model::Matrix values;
      for (auto& v : values) {
        v = r.real();
        if (check) {
          finite(v, "matrix value");
        }
      }

      return values;
    }

    void bounds(Reader& r, quint32 sampleCount) {
      r.ensureValues(quint64(std::max(1u, sampleCount)) * 6, 4);
      for (quint32 i = 0; i < std::max(1u, sampleCount); ++i) {
        double values[6];
        for (double& v : values) {
          v = r.real();
          finite(v, "legacy bounds");
        }

        for (int a = 0; a < 3; ++a) {
          if (values[a] > values[a + 3]) {
            fail("Inverted legacy bounds");
          }
        }
      }
    }

    void skipIndices(Reader& r) {
      const quint16 n = r.u16();
      r.take(quint64(n) * 2);
    }

    QVector<uint32_t> stripTriangles(const QVector<uint32_t>& strip, const Geometry& geometry) {
      QVector<uint32_t> result;
      for (qsizetype step = 0; step + 2 < strip.size(); ++step) {
        uint32_t a = strip[step], b = strip[step + 1], c = strip[step + 2];
        if (step & 1) {
          std::swap(a, b);
        }

        if (a == b || b == c || c == a) {
          continue;
        }

        const auto& pa  = geometry.vertices[a].position;
        const auto& pb  = geometry.vertices[b].position;
        const auto& pc  = geometry.vertices[c].position;
        const double ux = double(pb[0]) - pa[0], uy = double(pb[1]) - pa[1], uz = double(pb[2]) - pa[2];
        const double vx = double(pc[0]) - pa[0], vy = double(pc[1]) - pa[1], vz = double(pc[2]) - pa[2];
        if (uy * vz - uz * vy == 0 && uz * vx - ux * vz == 0 && ux * vy - uy * vx == 0) {
          continue;
        }

        result.append(a);
        result.append(b);
        result.append(c);
      }

      return result;
    }

    struct TextureIndex {
      QHash<QString, QStringList> files;

      explicit TextureIndex(const QString& root) {
        QDirIterator it(QDir(root).filePath("gameres/textures"), QDir::Files, QDirIterator::Subdirectories);
        while (it.hasNext()) {
          const auto path = it.next();
          const QFileInfo info(path);
          const auto ext = info.suffix().toCaseFolded();
          if (ext == "dds" || ext == "tga") {
            files[info.fileName().toCaseFolded()].append(info.absoluteFilePath());
          }
        }
      }

      QString resolve(const QString& reference, const QString& mesh, QStringList& warnings) const {
        QString local = reference;
        local.replace('\\', '/');
        const QString stem = QFileInfo(local).completeBaseName().toCaseFolded();
        for (const QString& ext : {QString(".dds"), QString(".tga")}) {
          const auto it = files.constFind(stem + ext);
          if (it == files.cend()) {
            continue;
          }

          if (it->size() == 1) {
            return it->front();
          }

          auto candidates = *it;
          candidates.sort();
          warnings.append(QString("Mesh %1: ambiguous albedo reference '%2': %3. Keeping the material untextured.").arg(mesh, reference, candidates.join(", ")));
          return {};
        }

        warnings.append(QString("Mesh %1: no local DDS/TGA for albedo reference '%2'. Keeping the material untextured.").arg(mesh, reference));
        return {};
      }
    };

    const TextureIndex& textureIndex(const QString& root) {
      QString key = QFileInfo(root).canonicalFilePath();
      if (key.isEmpty()) {
        key = QDir(root).absolutePath();
      }

      key = QDir::cleanPath(QDir::fromNativeSeparators(key));
#ifdef Q_OS_WIN
      key = key.toCaseFolded();
#endif
      static std::mutex mutex;
      static QHash<QString, std::shared_ptr<const TextureIndex>> indexes;
      const std::lock_guard lock(mutex);
      auto it = indexes.constFind(key);
      if (it == indexes.cend()) {
        it = indexes.insert(key, std::make_shared<const TextureIndex>(root));
      }

      return **it;
    }

    class Loader {
      Reader& r;
      Scene result;
      model::AnimationData animation;
      const TextureIndex& textures;
      bool hasSkeleton = false, anyVertex = false;
      quint64 triangleCount                = 0;
      quint32 renderedMeshCount            = 0;
      static constexpr float palette[5][3] = {{.76f, .79f, .82f}, {.68f, .73f, .77f}, {.78f, .74f, .68f}, {.66f, .73f, .71f}, {.74f, .70f, .75f}};

      void addInfluence(Vertex& vertex, quint32 matrixIndex, double weight) {
        if (!std::isfinite(weight) || weight < 0 || weight > std::numeric_limits<float>::max()) {
          fail("Mesh contains an invalid animation weight");
        }

        if (result.influences.size() >= std::numeric_limits<uint32_t>::max()) {
          fail("Too many model influences");
        }

        result.influences.append(QVector2D(float(matrixIndex), float(weight)));
        ++vertex.influenceCount;
      }

      void position(Vertex& v, int axis, double value) {
        finite(value, "vertex position");
        if (std::abs(value) > 1e9) {
          fail("Mesh exceeds the viewer's coordinate safety range (+/-1e+09 units)");
        }

        v.position[axis] = float(value);
      }

      void validateGeometry(const Geometry& g, const QVector<uint32_t>& indices) {
        for (uint32_t i : indices) {
          if (i >= quint64(g.vertices.size())) {
            fail("Index outside vertex array");
          }
        }
      }

      void addMesh(Geometry geometry, QVector<uint32_t> indices, const QString& material, const QString& reference, bool strip) {
        validateGeometry(geometry, indices);
        if (strip) {
          indices = stripTriangles(indices, geometry);
        } else if (indices.size() % 3) {
          fail("Mesh has an incomplete triangle list");
        }

        Mesh mesh;
        mesh.name            = QString("%1: %2").arg(result.meshes.size() + 1, 2, 10, QChar('0')).arg(material.isEmpty() ? "(unnamed material)" : material);
        mesh.geometry        = int(result.geometries.size());
        mesh.indices         = std::move(indices);
        mesh.material.name   = material;
        const QString albedo = textures.resolve(reference, mesh.name, result.warnings);
        if (albedo.isEmpty()) {
          mesh.material.albedo = "$white$";
          const auto& color    = palette[renderedMeshCount % 5];
          mesh.material.tint   = QVector3D(color[0], color[1], color[2]);
        } else {
          mesh.material.albedo     = albedo;
          mesh.material.flipAlbedo = QFileInfo(albedo).suffix().compare("tga", Qt::CaseInsensitive) == 0;
        }

        if (!mesh.indices.isEmpty()) {
          ++renderedMeshCount;
        }

        bool first = true;
        for (const auto& v : geometry.vertices) {
          const QVector3D point(v.position[0], v.position[1], v.position[2]);
          if (first) {
            mesh.bounds = {point, point};
            first       = false;
          }

          for (int a = 0; a < 3; ++a) {
            mesh.bounds.low[a]  = std::min(mesh.bounds.low[a], point[a]);
            mesh.bounds.high[a] = std::max(mesh.bounds.high[a], point[a]);
          }

          if (!anyVertex) {
            result.bounds = {point, point};
            anyVertex     = true;
          }

          for (int a = 0; a < 3; ++a) {
            result.bounds.low[a]  = std::min(result.bounds.low[a], point[a]);
            result.bounds.high[a] = std::max(result.bounds.high[a], point[a]);
          }
        }

        triangleCount += quint64(mesh.indices.size() / 3);
        result.geometries.append(std::move(geometry));
        result.meshes.append(std::move(mesh));
      }

      void readTracks(quint32 samples, quint32 count, double sampleRate, bool preMatrices) {
        if (!count) {
          fail("Empty sampled track palette");
        }

        r.ensureValues(quint64(count) * (quint64(samples) * 16 + (preMatrices ? 16 : 0)), 4);
        for (quint32 i = 0; i < count; ++i) {
          model::Track track;
          track.rate = sampleRate;
          if (preMatrices) {
            track.pre = matrix(r);
          }

          for (double value : track.pre) {
            if (std::abs(value) > std::numeric_limits<float>::max()) {
              fail("Invalid sampled pre-matrix values");
            }
          }

          track.samples.resize(samples);
          for (auto& sample : track.samples) {
            sample = matrix(r);
            for (double value : sample) {
              if (std::abs(value) > std::numeric_limits<float>::max()) {
                fail("Invalid sampled matrix values");
              }
            }
          }

          animation.tracks.append(std::move(track));
        }
      }

      void legacyMesh(double binaryRate) {
        const QString name     = r.string();
        const QString material = r.string();
        QString reference;
        for (int i = 0; i < 5; ++i) {
          const auto s = r.string();
          if (!i) {
            reference = s;
          }
        }

        const quint32 primitives = r.u32(), vertexCount = r.u32(), indexCount = r.u32();
        if (quint64(primitives) * 3 != indexCount) {
          fail("Inconsistent legacy triangle count");
        }

        zero(r.text ? r.integer() : r.u8(), "pre-vertex field");
        r.ensureValues(quint64(vertexCount) * 8, 4);
        Geometry geometry;
        geometry.vertices.resize(vertexCount);
        for (auto& v : geometry.vertices) {
          for (int a = 0; a < 3; ++a) {
            position(v, a, r.real());
          }

          for (float& n : v.normal) {
            const double value = r.real();
            finite(value, "vertex normal");
            n = float(value);
          }

          for (float& uv : v.uv) {
            const double value = r.real();
            finite(value, "vertex texture coordinate");
            uv = float(value);
          }
        }

        r.ensureValues(indexCount, 2);
        QVector<uint32_t> indices;
        indices.resize(indexCount);
        for (auto& i : indices) {
          if (r.text) {
            const auto value = r.integer();
            if (value < 0 || quint64(value) >= vertexCount) {
              fail("Index outside vertex array");
            }

            i = quint32(value);
          } else {
            i = r.u16();
          }
        }

        validateGeometry(geometry, indices);
        const quint32 samples = r.u32();
        double sampleRate     = binaryRate;
        quint32 mode          = 3;
        if (r.text) {
          sampleRate = rate(r.real());
          bounds(r, samples);
          zero(r.integer(), "text unknown_B");
          const qint64 hasMatrices = r.integer();
          if (hasMatrices != 0 && hasMatrices != 1) {
            fail("Unsupported text has_matrices field");
          }

          if (bool(samples) != bool(hasMatrices)) {
            fail("Unsupported text sample/matrix layout");
          }

          mode = hasMatrices ? 2 : 3;
        } else {
          mode = r.u8();
          if (mode < 1 || mode > 3) {
            fail("Unsupported legacy animation mode");
          }

          if ((mode == 3 && samples) || (mode != 3 && !samples)) {
            fail("Unsupported legacy mode/sample layout");
          }

          bounds(r, samples);
        }

        const quint32 trackBase = quint32(animation.tracks.size());
        if (mode == 1) {
          const quint32 trackCount = r.u16();
          readTracks(samples, trackCount, sampleRate, true);
          if (r.u32() != vertexCount) {
            fail("Assignment count differs from vertex count");
          }

          for (auto& v : geometry.vertices) {
            v.influenceStart = quint32(result.influences.size());
            const quint32 n  = r.u8();
            if (n >= 10) {
              fail("Mesh has more than nine influences per vertex");
            }

            for (quint32 i = 0; i < n; ++i) {
              const quint32 track = r.u16();
              const double weight = r.real();
              if (track >= trackCount) {
                fail("Track index outside palette");
              }

              addInfluence(v, trackBase + track, weight);
            }
          }
        } else if (mode == 2) {
          readTracks(samples, 1, sampleRate, false);
          for (auto& v : geometry.vertices) {
            v.influenceStart = quint32(result.influences.size());
            addInfluence(v, trackBase, 1);
          }
        }

        if (r.text) {
          zero(r.integer(), "text unknown_C");
        }

        addMesh(std::move(geometry), std::move(indices), material, reference, false);
      }

      void modernMesh() {
        const QString material = r.string();
        QString reference;
        for (int i = 0; i < 5; ++i) {
          const auto s = r.string();
          if (!i) {
            reference = s;
          }
        }

        r.take(24); // serialized bounds (display bounds are derived from vertices)
        r.u32();    // primitive count
        const bool strip = r.u8() != 0;
        r.u32(); // skinning count
        const quint32 indexCount = r.u32();
        r.ensureValues(indexCount, 2);
        QVector<uint32_t> indices;
        indices.resize(indexCount);
        for (auto& i : indices) {
          i = r.u16();
        }

        const quint32 count = r.u16();
        r.ensureValues(quint64(count) * 8, 4);
        Geometry geometry;
        geometry.vertices.resize(count);
        for (auto& v : geometry.vertices) {
          for (int a = 0; a < 3; ++a) {
            position(v, a, r.real());
          }
        }

        for (auto& v : geometry.vertices) {
          for (float& n : v.normal) {
            const double value = r.real();
            finite(value, "vertex normal");
            n = float(value);
          }
        }

        for (auto& v : geometry.vertices) {
          for (float& uv : v.uv) {
            uv = float(r.real());
          }
        }

        const bool assignments = r.u8() != 0;
        for (auto& v : geometry.vertices) {
          v.influenceStart = quint32(result.influences.size());
          if (!assignments) {
            continue;
          }

          const quint32 n = r.u8();
          if (n >= 10) {
            fail("Mesh has more than nine influences per vertex");
          }

          r.ensureValues(n, 6);
          for (quint32 i = 0; i < n; ++i) {
            const quint32 bone  = r.u16();
            const double weight = r.real();
            addInfluence(v, bone, weight);
          }
        }

        addMesh(std::move(geometry), std::move(indices), material, reference, strip);
      }

      void modernSkeleton() {
        hasSkeleton = r.u8() != 0;
        if (!hasSkeleton) {
          return;
        }

        const quint32 boneCount = r.u16();
        animation.bones.reserve(boneCount);
        for (quint32 i = 0; i < boneCount; ++i) {
          model::Bone bone;
          bone.name       = r.string();
          const auto disk = matrix(r);
          for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
              bone.rest[row * 4 + col] = disk[col * 4 + row];
            }
          }

          const quint16 parent = r.u16();
          bone.parent          = parent == 0xffff ? -1 : parent;
          skipIndices(r); // stored children; parent field is authoritative
          r.u16();        // animation slot
          animation.bones.append(std::move(bone));
        }

        skipIndices(r); // stored root bones
        const quint32 clipCount = r.u16();
        for (quint32 i = 0; i < clipCount; ++i) {
          model::Clip clip;
          clip.name                  = r.string();
          clip.duration              = r.real();
          clip.timeParameter         = r.real();
          const quint32 channelCount = r.u16();
          clip.channels.reserve(channelCount);
          for (quint32 c = 0; c < channelCount; ++c) {
            model::Channel channel;
            channel.bone = r.u16();
            r.string(); // duplicate bone name
            const quint32 keyCount = r.u32();
            r.ensureValues(keyCount, 45);
            channel.keys.resize(keyCount);
            for (auto& key : channel.keys) {
              key.flags = r.u8();
              for (double& v : key.rotation) {
                v = r.real();
              }

              for (double& v : key.translation) {
                v = r.real();
              }

              for (double& v : key.scale) {
                v = r.real();
              }

              key.time = r.real();
            }

            clip.channels.append(std::move(channel));
          }

          if (result.version == "1.0") {
            const quint32 events = r.u16();
            for (quint32 e = 0; e < events; ++e) {
              finite(r.real(), "animation event time");
              r.string();
            }
          }

          animation.clips.append(std::move(clip));
        }

        const quint32 groups = r.u16();
        for (quint32 i = 0; i < groups; ++i) {
          r.string();
          skipIndices(r);
          skipIndices(r);
        }

        r.u16(); // animation slot count
      }

    public:
      Loader(Reader& reader, const QString& path, const QString& root) : r(reader), textures(textureIndex(root)) {
        result.path = QFileInfo(path).absoluteFilePath();
        result.name = QFileInfo(path).completeBaseName();
      }

      Scene load() {
        if (r.string() != "DS2ModelFile_1") {
          fail("Invalid DS2ModelFile_1 signature");
        }

        result.version = r.string();
        if (r.text) {
          if (result.version != "0.5") {
            fail("Unsupported text DS2MD version " + result.version);
          }

          animation.sampled = result.sampledAnimation = true;
          const quint32 count                         = r.count();
          r.ensureValues(count, 1);
          for (quint32 i = 0; i < count; ++i) {
            legacyMesh(0);
          }

          zero(r.integer(), "text terminal");
        } else if (result.version == "0.6") {
          animation.sampled = result.sampledAnimation = true;
          const double sampleRate                     = rate(r.real());
          const quint32 count                         = r.u16();
          for (quint32 i = 0; i < count; ++i) {
            legacyMesh(sampleRate);
          }

          const quint32 collisions = r.u16();
          for (quint32 i = 0; i < collisions; ++i) {
            r.string();
            if (r.u8() != 1) {
              fail("Unsupported legacy collision type");
            }

            r.take(8);
            for (int j = 0; j < 3; ++j) {
              finite(r.real(), "collision parameter");
            }

            matrix(r);
          }
        } else if (result.version == "0.9" || result.version == "1.0") {
          const auto name = r.string();
          if (!name.isEmpty()) {
            result.name = name;
          }

          const quint32 count = r.u8();
          for (quint32 i = 0; i < count; ++i) {
            modernMesh();
          }

          const quint32 collisions = r.u8();
          for (quint32 i = 0; i < collisions; ++i) {
            r.string();
            const quint32 type = r.u8();
            r.u16();
            r.take(64);
            const quint32 params = type == 0 ? 1 : type == 1 ? 3 : type == 2 ? 2 : 0;
            r.take(params * 4);
          }

          modernSkeleton();
        } else {
          fail("Unsupported DS2MD version " + result.version + ". This viewer supports text 0.5 and standard binary 0.6, 0.9 and 1.0.");
        }

        r.end();
        if (!triangleCount) {
          fail("The model contains no renderable triangles");
        }

        const auto matrixCount = result.sampledAnimation ? animation.tracks.size() : animation.bones.size();
        for (const auto& entry : result.influences) {
          if (entry.x() < 0 || entry.x() >= matrixCount) {
            fail(QString("Mesh refers to missing animation matrix %1").arg(entry.x()));
          }
        }

        result.boneCount = int(matrixCount);
        result.note      = result.sampledAnimation ? "Stored geometry; camera fits the displayed pose." : "Bind-space geometry; camera fits the displayed pose.";
        if (result.sampledAnimation && !animation.tracks.isEmpty()) {
          model::Clip clip;
          clip.name = "Sampled animation";
          for (const auto& track : animation.tracks) {
            clip.duration = std::max(clip.duration, (track.samples.size() - 1) / track.rate);
          }

          finite(clip.duration, "sampled animation duration");
          animation.clips.append(std::move(clip));
          result.note += QString(" %1 sampled matrix tracks; 1 clips.").arg(matrixCount);
        } else if (hasSkeleton) {
          result.note += QString(" %1 bones; %2 animation clips.").arg(matrixCount).arg(animation.clips.size());
        }

        if (hasSkeleton || !animation.tracks.isEmpty()) {
          result.animation = model::makeAnimation(std::move(animation), result);
        }

        if (!result.animation) {
          result.influences.clear();
          for (auto& geometry : result.geometries) {
            for (auto& v : geometry.vertices) {
              v.influenceStart = v.influenceCount = 0;
            }
          }
        }

        return std::move(result);
      }
    };
  } // namespace

  Scene loadModel(const QString& path, const QString& gameRoot) {
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
      fail(QString("Cannot open model %1: %2").arg(path, file.errorString()));
    }

    const QByteArray bytes = file.readAll();
    if (file.error() != QFileDevice::NoError) {
      fail(QString("Cannot read model %1: %2").arg(path, file.errorString()));
    }

    if (bytes.startsWith("DSMO")) {
      fail("Unsupported DS2MD version DSMO packed/cache. This viewer supports text 0.5 and standard binary "
           "0.6, 0.9 and 1.0.");
    }

    Reader reader(bytes);
    reader.whitespace();
    reader.text = reader.remaining() && bytes[reader.pos] == '"';
    if (!reader.text) {
      reader.pos = 0;
    }

    return Loader(reader, path, gameRoot).load();
  }
} // namespace ds2
