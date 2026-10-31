#include "model_animation.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace ds2::model {
  namespace {
    [[noreturn]] void fail(const QString& message) {
      throw std::runtime_error(message.toStdString());
    }

    void finite(double value, const QString& label) {
      if (!std::isfinite(value)) {
        fail(label + " must be finite");
      }
    }

    void finite(double value, const char* label) {
      if (!std::isfinite(value)) {
        fail(QString::fromLatin1(label) + " must be finite");
      }
    }

    Matrix multiply(const Matrix& a, const Matrix& b) {
      Matrix result{};
      for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
          for (int k = 0; k < 4; ++k) {
            result[row * 4 + col] += a[row * 4 + k] * b[k * 4 + col];
          }
        }
      }

      return result;
    }

    void finiteMatrix(const Matrix& m, const QString& label, bool f32 = false) {
      for (double v : m) {
        if (!std::isfinite(v) || (f32 && std::abs(v) > std::numeric_limits<float>::max())) {
          fail(label + " contains non-finite or unrepresentable matrix values");
        }
      }
    }

    void checkPoseMatrix(const Matrix& m, const char* kind, qsizetype index, bool f32 = false) {
      for (double v : m) {
        if (!std::isfinite(v) || (f32 && std::abs(v) > std::numeric_limits<float>::max())) {
          fail(QString("%1 %2 matrix is not finite or representable as f32").arg(QString::fromLatin1(kind)).arg(index));
        }
      }
    }

    Matrix inverse(const Matrix& m, const QString& label) {
      double rows[4][8]{};
      for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
          rows[r][c] = m[r * 4 + c];
        }

        rows[r][r + 4] = 1;
      }

      for (int c = 0; c < 4; ++c) {
        int pivot = c;
        for (int r = c + 1; r < 4; ++r) {
          if (std::abs(rows[r][c]) > std::abs(rows[pivot][c])) {
            pivot = r;
          }
        }

        if (rows[pivot][c] == 0) {
          fail(label + " has a singular bind matrix");
        }

        for (int k = 0; k < 8; ++k) {
          std::swap(rows[c][k], rows[pivot][k]);
        }

        const double divisor = rows[c][c];
        for (double& v : rows[c]) {
          v /= divisor;
        }

        for (int r = 0; r < 4; ++r) {
          if (r == c) {
            continue;
          }

          const double factor = rows[r][c];
          for (int k = 0; k < 8; ++k) {
            rows[r][k] -= factor * rows[c][k];
          }
        }
      }

      Matrix result;
      for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
          result[r * 4 + c] = rows[r][c + 4];
        }
      }

      finiteMatrix(result, label);
      return result;
    }

    QMatrix4x4 qtMatrix(const Matrix& m, bool diskOrder = false) {
      QMatrix4x4 result;
      for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
          result(r, c) = float(m[diskOrder ? c * 4 + r : r * 4 + c]);
        }
      }

      return result;
    }

    using Quaternion = std::array<double, 4>;

    Quaternion normalized(Quaternion q) {
      const double norm = std::hypot(std::hypot(q[0], q[1]), std::hypot(q[2], q[3]));
      if (norm == 0) {
        fail("Animation key contains a zero quaternion");
      }

      for (double& v : q) {
        v /= norm;
      }

      return q;
    }

    Quaternion slerp(const Quaternion& a, Quaternion b, double fraction) {
      double dot = 0;
      for (int i = 0; i < 4; ++i) {
        dot += a[i] * b[i];
      }

      if (dot < 0) {
        for (double& v : b) {
          v = -v;
        }

        dot = -dot;
      }

      dot         = std::min(1.0, dot);
      double left = 1 - fraction, right = fraction;
      if (dot <= 0.9995) {
        const double angle = std::acos(dot), divisor = std::sin(angle);
        left  = std::sin((1 - fraction) * angle) / divisor;
        right = std::sin(fraction * angle) / divisor;
      }

      Quaternion q;
      for (int i = 0; i < 4; ++i) {
        q[i] = a[i] * left + b[i] * right;
      }

      return normalized(q);
    }

    Matrix trs(const Quaternion& q, const std::array<double, 3>& t, const std::array<double, 3>& s) {
      const auto [x, y, z, w] = q;
      return {(1 - 2 * (y * y + z * z)) * s[0],
              2 * (x * y - w * z) * s[0],
              2 * (x * z + w * y) * s[0],
              t[0],
              2 * (x * y + w * z) * s[1],
              (1 - 2 * (x * x + z * z)) * s[1],
              2 * (y * z - w * x) * s[1],
              t[1],
              2 * (x * z - w * y) * s[2],
              2 * (y * z + w * x) * s[2],
              (1 - 2 * (x * x + y * y)) * s[2],
              t[2],
              0,
              0,
              0,
              1};
    }

    Matrix sample(const Channel& channel, double time) {
      const auto& keys = channel.keys;
      auto it          = std::lower_bound(keys.cbegin(), keys.cend(), time, [](const Key& key, double t) {
        return key.time < t;
      });
      if (it == keys.cend()) {
        --it;
      }

      const Key& b = *it;
      constexpr std::array<double, 3> unit{1, 1, 1};
      if (it == keys.cbegin() || time >= keys.back().time || time == b.time) {
        return trs(b.rotation, b.translation, (b.flags & 8) ? b.scale : unit);
      }

      const Key& a       = *(it - 1);
      const double f     = (time - a.time) / (b.time - a.time);
      const Quaternion q = (b.flags & 1) ? a.rotation : slerp(a.rotation, b.rotation, f);
      auto t = a.translation, s = unit;
      for (int i = 0; i < 3; ++i) {
        if (!(b.flags & 2)) {
          t[i] += (b.translation[i] - t[i]) * f;
        }

        if (b.flags & 8) {
          if (b.flags & 4) {
            s[i] = (a.flags & 8) ? a.scale[i] : 1;
          } else {
            s[i] = a.scale[i] + (b.scale[i] - a.scale[i]) * f;
          }
        }
      }

      return trs(q, t, s);
    }

    struct Box {
      bool valid = false;
      std::array<double, 3> low{}, high{};

      void include(const std::array<double, 3>& p) {
        if (!valid) {
          low = high = p;
          valid      = true;
          return;
        }

        for (int a = 0; a < 3; ++a) {
          low[a]  = std::min(low[a], p[a]);
          high[a] = std::max(high[a], p[a]);
        }
      }

      Bounds bounds() const {
        return {{float(low[0]), float(low[1]), float(low[2])}, {float(high[0]), float(high[1]), float(high[2])}};
      }
    };

    class Player final : public Animation {
      AnimationData data;
      QVector<int> order;
      QVector<Matrix> inverseBind, local, global;
      QVector<QMatrix4x4> palette, nextPalette;
      QVector<bool> decoded;
      QVector<Box> boneBounds;
      Box unweighted;
      Bounds fallback;
      double weightMin = std::numeric_limits<double>::infinity(), weightMax = 0;
      int selected   = -1;
      double seconds = 0;

      void decode(int index) {
        if (decoded[index]) {
          return;
        }

        auto& clip = data.clips[index];
        for (auto& channel : clip.channels) {
          double previous = -std::numeric_limits<double>::infinity();
          for (auto& key : channel.keys) {
            const QString label = QString("Clip %1, bone %2 key").arg(clip.name).arg(channel.bone);
            finite(key.time, label);
            if (key.time <= previous) {
              fail(label + " times must be strictly increasing");
            }

            previous = key.time;
            for (double v : key.rotation) {
              finite(v, label);
            }

            for (double v : key.translation) {
              finite(v, label);
            }

            for (double v : key.scale) {
              finite(v, label);
            }

            key.rotation = normalized(key.rotation);
          }
        }

        decoded[index] = true;
      }

      void evaluate(int index, double time) {
        if (data.sampled) {
          for (qsizetype i = 0; i < data.tracks.size(); ++i) {
            const auto& track     = data.tracks[i];
            const qsizetype last  = track.samples.size() - 1;
            const qsizetype frame = time >= last / track.rate ? last : qsizetype(std::nextafter(time * track.rate, std::numeric_limits<double>::infinity()));
            const Matrix m        = multiply(track.pre, track.samples[frame]);
            checkPoseMatrix(m, "Sampled track", i, true);
            nextPalette[i] = qtMatrix(m, true);
          }

          palette.swap(nextPalette);
          return;
        }

        for (qsizetype i = 0; i < data.bones.size(); ++i) {
          local[i] = data.bones[i].rest;
        }

        for (const auto& channel : data.clips[index].channels) {
          if (!channel.keys.isEmpty()) {
            local[channel.bone] = sample(channel, time);
          }
        }

        for (int i : order) {
          const int parent = data.bones[i].parent;
          global[i]        = parent == -1 ? local[i] : multiply(global[parent], local[i]);
          checkPoseMatrix(global[i], "Animated bone", i);
          const Matrix skin = multiply(global[i], inverseBind[i]);
          checkPoseMatrix(skin, "Animated bone skin", i, true);
          nextPalette[i] = qtMatrix(skin);
        }

        palette.swap(nextPalette);
      }

    public:
      Player(AnimationData source, const Scene& scene) : data(std::move(source)), fallback(scene.bounds) {
        const qsizetype count = data.sampled ? data.tracks.size() : data.bones.size();
        palette.resize(count);
        nextPalette.resize(count);
        boneBounds.resize(count);
        decoded.fill(false, data.clips.size());
        if (!data.sampled) {
          QVector<QVector<int>> children(count);
          QVector<int> pending;
          inverseBind.resize(count);
          local.resize(count);
          global.resize(count);
          for (qsizetype i = 0; i < count; ++i) {
            const auto& bone = data.bones[i];
            finiteMatrix(bone.rest, QString("Bone %1 rest").arg(i));
            if (bone.parent < -1 || bone.parent >= count) {
              fail(QString("Bone %1 has invalid parent").arg(i));
            }

            if (bone.parent == -1) {
              pending.append(int(i));
            } else {
              children[bone.parent].append(int(i));
            }
          }

          while (!pending.isEmpty()) {
            const int i = pending.takeLast();
            order.append(i);
            for (int child : children[i]) {
              pending.append(child);
            }
          }

          if (order.size() != count) {
            fail("Skeleton parent hierarchy contains a cycle");
          }

          for (int i : order) {
            const auto& bone = data.bones[i];
            global[i]        = bone.parent == -1 ? bone.rest : multiply(global[bone.parent], bone.rest);
            finiteMatrix(global[i], QString("Bone %1 bind").arg(i));
            inverseBind[i] = inverse(global[i], QString("Bone %1").arg(i));
          }
        }

        for (const auto& clip : data.clips) {
          finite(clip.duration, "Clip duration");
          if (clip.duration < 0) {
            fail("Clip duration must be nonnegative");
          }

          finite(clip.timeParameter, "Clip time parameter");
          QVector<bool> seen(count, false);
          for (const auto& channel : clip.channels) {
            if (channel.bone < 0 || channel.bone >= count) {
              fail("Animation channel refers to a missing bone");
            }

            if (seen[channel.bone]) {
              fail("Animation clip has duplicate channels for a bone");
            }

            seen[channel.bone] = true;
          }
        }

        for (const auto& geometry : scene.geometries) {
          for (const auto& vertex : geometry.vertices) {
            const std::array<double, 3> point{vertex.position[0], vertex.position[1], vertex.position[2]};
            if (!count || !vertex.influenceCount) {
              unweighted.include(point);
              continue;
            }

            double total = 0;
            for (uint32_t i = 0; i < vertex.influenceCount; ++i) {
              const auto& entry = scene.influences[vertex.influenceStart + i];
              total += entry.y();
              boneBounds[int(entry.x())].include(point);
            }

            weightMin = std::min(weightMin, total);
            weightMax = std::max(weightMax, total);
          }
        }

        if (!data.clips.isEmpty()) {
          select(0);
        }
      }

      int clipCount() const override {
        return int(data.clips.size());
      }

      QString clipName(int index) const override {
        if (index < 0 || index >= data.clips.size()) {
          fail("Invalid animation clip index");
        }

        return data.clips[index].name;
      }

      int clipIndex() const override {
        return selected;
      }

      double time() const override {
        return seconds;
      }

      double duration() const override {
        return selected < 0 ? 0 : data.clips[selected].duration;
      }

      void select(int index) override {
        if (index == -1) {
          for (auto& m : palette) {
            m.setToIdentity();
          }

          selected = -1;
          seconds  = 0;
          playing  = false;
          return;
        }

        if (index < 0 || index >= data.clips.size()) {
          fail("Invalid animation clip index");
        }

        decode(index);
        evaluate(index, 0);
        selected = index;
        seconds  = 0;
        playing  = true;
      }

      void seek(double value) override {
        finite(value, "Seek time");
        const double t = std::clamp(value, 0.0, duration());
        if (selected >= 0) {
          evaluate(selected, t);
        }

        seconds = t;
      }

      bool update(double dt) override {
        if (!playing || selected < 0) {
          return false;
        }

        finite(dt, "Animation delta time");
        if (dt < 0) {
          fail("Animation delta time must be nonnegative");
        }

        finite(speed, "Animation speed");
        const double delta = dt * speed;
        finite(delta, "Animation time advance");
        const double d = duration();
        if (d == 0) {
          playing = false;
          return false;
        }

        if (delta == 0) {
          return false;
        }

        double t = seconds + delta;
        finite(t, "Animation time");
        bool stop = false;
        if (loop) {
          if (t < 0 || t >= d) {
            t = std::fmod(t, d);
            if (t < 0) {
              t += d;
            }
          }
        } else if (t <= 0 && delta < 0) {
          t    = 0;
          stop = true;
        } else if (t >= d && delta > 0) {
          t    = d;
          stop = true;
        }

        const bool changed = t != seconds;
        if (changed) {
          evaluate(selected, t);
          seconds = t;
        }

        if (stop) {
          playing = false;
        }

        return changed;
      }

      const QVector<QMatrix4x4>& matrices() const override {
        return palette;
      }

      const AnimationData& sourceData() const override {
        return data;
      }

      Bounds bounds() const override {
        Box weighted, result;
        for (qsizetype i = 0; i < boneBounds.size(); ++i) {
          const auto& box = boneBounds[i];
          if (!box.valid) {
            continue;
          }

          const auto& m = palette[i];
          for (int corner = 0; corner < 8; ++corner) {
            const double x = (corner & 1) ? box.high[0] : box.low[0];
            const double y = (corner & 2) ? box.high[1] : box.low[1];
            const double z = (corner & 4) ? box.high[2] : box.low[2];
            std::array<double, 3> point;
            for (int row = 0; row < 3; ++row) {
              point[row] = double(m(row, 0)) * x + double(m(row, 1)) * y + double(m(row, 2)) * z + m(row, 3);
            }

            weighted.include(point);
          }
        }

        if (weighted.valid) {
          result.valid = true;
          for (int a = 0; a < 3; ++a) {
            const std::array<double, 4> values{weighted.low[a] * weightMin, weighted.low[a] * weightMax, weighted.high[a] * weightMin, weighted.high[a] * weightMax};
            result.low[a]  = *std::min_element(values.begin(), values.end());
            result.high[a] = *std::max_element(values.begin(), values.end());
          }
        }

        if (unweighted.valid) {
          result.include(unweighted.low);
          result.include(unweighted.high);
        }

        return result.valid ? result.bounds() : fallback;
      }
    };
  } // namespace

  Matrix identity() {
    return {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
  }

  std::shared_ptr<Animation> makeAnimation(AnimationData data, const Scene& scene) {
    return std::make_shared<Player>(std::move(data), scene);
  }
} // namespace ds2::model
