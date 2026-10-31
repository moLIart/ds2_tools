#include "glb_export.h"
#include "model_animation.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ds2::glb {
namespace {
using model::Matrix;
using Vec3 = std::array<double, 3>;
using Quat = std::array<double, 4>;
using Mat3 = std::array<Vec3, 3>;
constexpr Quat identityRotation{0, 0, 0, 1};
constexpr Vec3 unitScale{1, 1, 1};

[[noreturn]] void fail(const QString &message) {
    throw std::runtime_error(("GLB animation: " + message).toStdString());
}
float real(double value, const QString &label) {
    if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<float>::max())
        fail(label + " is not finite or representable as float32");

    return float(value);
}
void affine(const Matrix &m, const QString &label) {
    for (double v : m)
        if (!std::isfinite(v))
            fail(label + " contains a non-finite matrix value");

    if (m[12] != 0 || m[13] != 0 || m[14] != 0 || m[15] != 1)
        fail(label + " is non-affine (expected final row [0, 0, 0, 1])");
}
void diskAffine(const Matrix &m, const QString &label) {
    for (double v : m)
        if (!std::isfinite(v))
            fail(label + " contains a non-finite matrix value");

    if (m[3] != 0 || m[7] != 0 || m[11] != 0 || m[15] != 1)
        fail(label + " is non-affine (expected disk final column [0, 0, 0, 1])");
}
Matrix multiply(const Matrix &a, const Matrix &b) {
    Matrix result{};
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            for (int k = 0; k < 4; ++k)
                result[r * 4 + c] += a[r * 4 + k] * b[k * 4 + c];

    return result;
}
Matrix inverse(const Matrix &m, const QString &label) {
    double rows[4][8]{};
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c)
            rows[r][c] = m[r * 4 + c];

        rows[r][r + 4] = 1;
    }

    for (int c = 0; c < 4; ++c) {
        int pivot = c;
        for (int r = c + 1; r < 4; ++r)
            if (std::abs(rows[r][c]) > std::abs(rows[pivot][c]))
                pivot = r;

        if (rows[pivot][c] == 0)
            fail(label + " has a singular cumulative rest matrix");

        for (int k = 0; k < 8; ++k)
            std::swap(rows[pivot][k], rows[c][k]);

        const double divisor = rows[c][c];
        for (double &v : rows[c])
            v /= divisor;

        for (int r = 0; r < 4; ++r) {
            if (r == c)
                continue;

            const double factor = rows[r][c];
            for (int k = 0; k < 8; ++k)
                rows[r][k] -= factor * rows[c][k];
        }
    }

    Matrix result{};
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            result[r * 4 + c] = rows[r][c + 4];

    return result;
}
double dot(const Vec3 &a, const Vec3 &b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
Vec3 cross(const Vec3 &a, const Vec3 &b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]};
}
double norm(const Vec3 &v) {
    return std::hypot(v[0], v[1], v[2]);
}
Quat normalized(Quat q, const QString &label) {
    const double n = std::hypot(std::hypot(q[0], q[1]), std::hypot(q[2], q[3]));
    if (!std::isfinite(n) || n == 0)
        fail(label + " has an invalid/zero quaternion");

    for (double &v : q)
        v /= n;

    return q;
}
Quat quaternion(const Mat3 &columns, const QString &label) {
    const auto at = [&](int r, int c) { return columns[c][r]; };
    Quat q{};
    const double trace = at(0, 0) + at(1, 1) + at(2, 2);
    if (trace > 0) {
        const double s = 2 * std::sqrt(1 + trace);
        q = {(at(2, 1) - at(1, 2)) / s, (at(0, 2) - at(2, 0)) / s,
             (at(1, 0) - at(0, 1)) / s, s / 4};
    } else {
        int i = 0;
        if (at(1, 1) > at(i, i))
            i = 1;

        if (at(2, 2) > at(i, i))
            i = 2;

        const int j = (i + 1) % 3, k = (i + 2) % 3;
        const double s = 2 * std::sqrt(std::max(0.0, 1 + at(i, i) - at(j, j) - at(k, k)));
        q[i] = s / 4;
        q[j] = (at(j, i) + at(i, j)) / s;
        q[k] = (at(k, i) + at(i, k)) / s;
        q[3] = (at(k, j) - at(j, k)) / s;
    }

    return normalized(q, label);
}
struct Factors {
    Vec3 translation{}, scale = unitScale;
    Quat left = identityRotation, right = identityRotation;
};
// One-sided Jacobi SVD avoids squaring A's condition number. Columns are used
// throughout: A = U * diag(s) * transpose(V). Reflection signs belong to s,
// leaving both U and transpose(V) proper rotations suitable for quaternions.
Factors factor(const Matrix &m, const QString &label) {
    affine(m, label);
    Factors result;
    result.translation = {m[3], m[7], m[11]};
    double magnitude = 0;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            magnitude = std::max(magnitude, std::abs(m[r * 4 + c]));

    if (magnitude == 0) {
        result.scale = {};
        return result;
    }

    Mat3 b{}, v{{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
    for (int c = 0; c < 3; ++c)
        for (int r = 0; r < 3; ++r)
            b[c][r] = m[r * 4 + c] / magnitude;

    constexpr double epsilon = std::numeric_limits<double>::epsilon();
    bool converged = false;
    for (int sweep = 0; sweep < 64; ++sweep) {
        bool changed = false;
        for (int p = 0; p < 2; ++p) {
            for (int q = p + 1; q < 3; ++q) {
                const double alpha = dot(b[p], b[p]), beta = dot(b[q], b[q]);
                const double gamma = dot(b[p], b[q]);
                if (std::abs(gamma) <= 8 * epsilon * std::sqrt(alpha) * std::sqrt(beta))
                    continue;

                const double zeta = (beta - alpha) / (2 * gamma);
                const double t = std::copysign(1.0, zeta) / (std::abs(zeta) + std::hypot(1.0, zeta));
                const double c = 1 / std::hypot(1.0, t), s = c * t;
                for (int r = 0; r < 3; ++r) {
                    const double bp = b[p][r], vp = v[p][r];
                    b[p][r] = c * bp - s * b[q][r];
                    b[q][r] = s * bp + c * b[q][r];
                    v[p][r] = c * vp - s * v[q][r];
                    v[q][r] = s * vp + c * v[q][r];
                }

                changed = true;
            }
        }

        if (!changed) {
            converged = true;
            break;
        }
    }

    if (!converged)
        fail(label + " affine SVD failed to converge");

    // Order nonzero columns before null-space completion; retain even very small
    // nonzero singular values rather than flattening highly anisotropic poses.
    for (int p = 0; p < 2; ++p)
        for (int q = p + 1; q < 3; ++q)
            if (norm(b[q]) > norm(b[p])) {
                std::swap(b[p], b[q]);
                std::swap(v[p], v[q]);
            }

    Mat3 u{};
    for (int c = 0; c < 3; ++c) {
        const double length = norm(b[c]);
        if (length > 0) {
            for (int r = 0; r < 3; ++r)
                u[c][r] = b[c][r] / length;

            result.scale[c] = length * magnitude;
        } else {
            // Any orthonormal completion is equivalent for a zero singular value.
            Vec3 best{};
            double bestLength = -1;
            for (int axis = 0; axis < 3; ++axis) {
                Vec3 candidate{};
                candidate[axis] = 1;
                for (int j = 0; j < c; ++j) {
                    const double projection = dot(candidate, u[j]);
                    for (int r = 0; r < 3; ++r)
                        candidate[r] -= projection * u[j][r];
                }

                if (norm(candidate) > bestLength) {
                    best = candidate;
                    bestLength = norm(candidate);
                }
            }

            for (int r = 0; r < 3; ++r)
                u[c][r] = best[r] / bestLength;

            result.scale[c] = 0;
        }
    }

    if (dot(cross(u[0], u[1]), u[2]) < 0) {
        for (double &x : u[2])
            x = -x;

        result.scale[2] = -result.scale[2];
    }

    if (dot(cross(v[0], v[1]), v[2]) < 0) {
        for (double &x : v[2])
            x = -x;

        result.scale[2] = -result.scale[2];
    }

    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            double reconstructed = 0;
            for (int k = 0; k < 3; ++k)
                reconstructed += u[k][r] * (result.scale[k] / magnitude) * v[k][c];

            if (std::abs(reconstructed - m[r * 4 + c] / magnitude) > 1e-11)
                fail(label + " affine SVD reconstruction failed");
        }
    }

    Mat3 transposeV{};
    for (int c = 0; c < 3; ++c)
        for (int r = 0; r < 3; ++r)
            transposeV[c][r] = v[r][c];

    result.left = quaternion(u, label);
    result.right = quaternion(transposeV, label);
    return result;
}
template <size_t N> QJsonArray json(const std::array<double, N> &values, const QString &label) {
    QJsonArray result;
    for (double value : values)
        result.append(double(real(value, label)));

    return result;
}
template <size_t N> void append(QVector<float> &output, const std::array<double, N> &values,
                               const QString &label) {
    for (double value : values)
        output.append(real(value, label));
}
void appendRotation(QVector<float> &output, Quat q, const QString &label) {
    if (!output.isEmpty()) {
        double d = 0;
        for (int i = 0; i < 4; ++i)
            d += double(output[output.size() - 4 + i]) * q[i];

        if (d < 0)
            for (double &v : q)
                v = -v;
    }

    append(output, q, label);
}
struct Joint {
    int helper = -1, joint = -1;
};
Joint addJoint(Document &document, const QString &name, const Factors &factors) {
    const int first = int(document.nodes.size());
    document.nodes.append(QJsonObject{{"name", name + " / affine"},
                                      {"translation", json(factors.translation, name)},
                                      {"rotation", json(factors.left, name)},
                                      {"scale", json(factors.scale, name)}});
    document.nodes.append(QJsonObject{{"name", name}, {"rotation", json(factors.right, name)}});
    document.child(first, first + 1);
    return {first, first + 1};
}
void appendMatrix(QVector<float> &output, const Matrix &m, const QString &label) {
    for (int c = 0; c < 4; ++c)
        for (int r = 0; r < 4; ++r)
            output.append(real(m[r * 4 + c], label));
}
struct AnimationBuilder {
    Document &document;
    QJsonArray channels, samplers;
    int input(const QVector<float> &times) {
        return document.floats(std::span<const float>(times.constData(), size_t(times.size())), "SCALAR", true);
    }
    void channel(int times, const QVector<float> &values, const char *type, int node,
                 const char *path, const char *interpolation = "LINEAR") {
        const int sampler = int(samplers.size());
        const int output = document.floats(std::span<const float>(values.constData(), size_t(values.size())), type);
        samplers.append(QJsonObject{{"input", times}, {"output", output}, {"interpolation", interpolation}});
        channels.append(QJsonObject{{"sampler", sampler},
                                     {"target", QJsonObject{{"node", node}, {"path", path}}}});
    }
    void finish(const model::Clip &clip, int identityNode) {
        // glTF requires nonempty channels/samplers. A constant identity target
        // preserves even empty clips and their duration without moving geometry.
        if (channels.isEmpty()) {
            QVector<float> times{0};
            const float end = real(clip.duration, "Clip duration");
            if (end > 0)
                times.append(end);

            QVector<float> values(times.size() * 3, 0);
            channel(input(times), values, "VEC3", identityNode, "translation");
        }

        document.animations.append(QJsonObject{
            {"name", clip.name}, {"channels", channels}, {"samplers", samplers},
            {"extras", QJsonObject{{"duration", clip.duration}, {"timeParameter", clip.timeParameter}}}});
    }
};
Quat slerp(const Quat &a, Quat b, double fraction, const QString &label) {
    double d = 0;
    for (int i = 0; i < 4; ++i)
        d += a[i] * b[i];

    if (d < 0) {
        for (double &v : b)
            v = -v;

        d = -d;
    }

    d = std::min(1.0, d);
    double left = 1 - fraction, right = fraction;
    if (d <= 0.9995) {
        const double angle = std::acos(d), divisor = std::sin(angle);
        left = std::sin((1 - fraction) * angle) / divisor;
        right = std::sin(fraction * angle) / divisor;
    }

    Quat q{};
    for (int i = 0; i < 4; ++i)
        q[i] = a[i] * left + b[i] * right;

    return normalized(q, label);
}
struct Pose {
    Vec3 translation{}, scale = unitScale;
    Quat rotation = identityRotation;
};
Pose sample(const model::Channel &channel, const QVector<Quat> &rotations, double time,
            const QString &label) {
    const auto &keys = channel.keys;
    auto it = std::lower_bound(keys.cbegin(), keys.cend(), time,
                              [](const model::Key &key, double t) { return key.time < t; });
    if (it == keys.cend())
        --it;

    const qsizetype index = it - keys.cbegin();
    const auto &b = *it;
    if (it == keys.cbegin() || time >= keys.back().time || time == b.time)
        return {b.translation, (b.flags & 8) ? b.scale : unitScale, rotations[index]};

    const auto &a = *(it - 1);
    const double fraction = (time - a.time) / (b.time - a.time);
    Pose result{a.translation, unitScale, (b.flags & 1) ? rotations[index - 1]
                                                      : slerp(rotations[index - 1], rotations[index], fraction, label)};
    for (int i = 0; i < 3; ++i) {
        if (!(b.flags & 2))
            result.translation[i] += (b.translation[i] - result.translation[i]) * fraction;

        if (b.flags & 8) {
            if (b.flags & 4)
                result.scale[i] = (a.flags & 8) ? a.scale[i] : 1;
            else
                result.scale[i] = a.scale[i] + (b.scale[i] - a.scale[i]) * fraction;
        }
    }

    return result;
}
bool differentRotation(const Quat &a, const Quat &b) {
    // Compare both signs: q and -q encode the same rotation.
    double same = 0, opposite = 0;
    for (int i = 0; i < 4; ++i) {
        same = std::max(same, std::abs(a[i] - b[i]));
        opposite = std::max(opposite, std::abs(a[i] + b[i]));
    }

    return std::min(same, opposite) > 1e-14;
}
QVector<Quat> validateKeys(const model::Channel &channel, const QString &label) {
    QVector<Quat> rotations;
    rotations.reserve(channel.keys.size());
    double previous = -std::numeric_limits<double>::infinity();
    float previousFloat = -std::numeric_limits<float>::infinity();
    for (const auto &key : channel.keys) {
        const float time = real(key.time, label + " key time");
        if (key.time <= previous)
            fail(label + " key times are not strictly increasing");

        if (time <= previousFloat)
            fail(label + " key times collide at float32 precision");

        // Modern disk times are float32 already. Reject instead of silently
        // moving a discontinuity if a future source supplies finer times.
        if (double(time) != key.time)
            fail(label + " key time is not exactly representable as float32");

        previous = key.time;
        previousFloat = time;
        for (double value : key.translation)
            real(value, label + " translation");

        for (double value : key.scale)
            real(value, label + " scale");

        for (double value : key.rotation)
            real(value, label + " rotation");

        rotations.append(normalized(key.rotation, label));
    }

    return rotations;
}
void modernChannel(AnimationBuilder &builder, const model::Channel &channel, const Joint &joint,
                   double duration, const QString &label, bool &discontinuities) {
    const QVector<Quat> rotations = validateKeys(channel, label);
    if (channel.keys.isEmpty())
        return;

    const float end = real(duration, label + " duration");
    QVector<float> times{0};
    if (end > 0)
        times.append(end);

    const auto addBoundary = [&](double keyTime, bool before, bool after) {
        if (keyTime < 0 || keyTime > duration)
            return;

        const float t = float(keyTime);
        times.append(t);
        if (before && t > 0) {
            const float previous = std::nextafter(t, -std::numeric_limits<float>::infinity());
            if (previous >= 0 && previous <= end)
                times.append(previous);

            discontinuities = true;
        }

        if (after && t < end) {
            const float next = std::nextafter(t, std::numeric_limits<float>::infinity());
            if (next <= end)
                times.append(next);

            discontinuities = true;
        }
    };
    for (qsizetype i = 0; i < channel.keys.size(); ++i) {
        const auto &b = channel.keys[i];
        const Vec3 exactScale = (b.flags & 8) ? b.scale : unitScale;
        bool before = false, after = false;
        if (i > 0) {
            const auto &a = channel.keys[i - 1];
            Vec3 leftScale = unitScale;
            if (b.flags & 8)
                leftScale = (b.flags & 4) ? ((a.flags & 8) ? a.scale : unitScale) : b.scale;

            before = ((b.flags & 1) && differentRotation(rotations[i - 1], rotations[i])) ||
                     ((b.flags & 2) && a.translation != b.translation) || leftScale != exactScale;
        }

        if (i + 1 < channel.keys.size()) {
            const auto &right = channel.keys[i + 1];
            Vec3 rightScale = unitScale;
            if (right.flags & 8)
                rightScale = (right.flags & 4) ? exactScale : b.scale;

            after = rightScale != exactScale;
        }

        addBoundary(b.time, before, after);
    }

    std::sort(times.begin(), times.end());
    times.erase(std::unique(times.begin(), times.end()), times.end());
    QVector<float> translations, scales, quaternions;
    translations.reserve(times.size() * 3);
    scales.reserve(times.size() * 3);
    quaternions.reserve(times.size() * 4);
    for (float time : times) {
        const Pose pose = sample(channel, rotations, double(time), label);
        append(translations, pose.translation, label);
        append(scales, pose.scale, label);
        appendRotation(quaternions, pose.rotation, label);
    }

    const int input = builder.input(times);
    builder.channel(input, translations, "VEC3", joint.helper, "translation");
    builder.channel(input, scales, "VEC3", joint.helper, "scale");
    builder.channel(input, quaternions, "VEC4", joint.joint, "rotation");
    // A channel replaces the entire local rest matrix, not just its translation
    // and scale. Explicitly remove the SVD's rest U rotation for this clip.
    QVector<float> endpoints{0};
    if (end > 0)
        endpoints.append(end);

    QVector<float> identity;
    for (qsizetype i = 0; i < endpoints.size(); ++i)
        append(identity, identityRotation, label);

    builder.channel(builder.input(endpoints), identity, "VEC4", joint.helper, "rotation");
}
void sampledTrack(AnimationBuilder &builder, const model::Track &track, const Joint &joint,
                  double duration, const QString &label) {
    if (!std::isfinite(track.rate) || track.rate <= 0 || track.samples.isEmpty())
        fail(label + " has an invalid sample rate or no samples");

    // Validate/factor every stored sample, including samples outside this clip.
    diskAffine(track.pre, label + " pre-matrix");
    QVector<float> times;
    QVector<float> translations, scales, left, right;
    float previous = -1;
    Factors last;
    bool haveLast = false;
    for (qsizetype frame = 0; frame < track.samples.size(); ++frame) {
        diskAffine(track.samples[frame], label + " disk sample");
        const Matrix disk = multiply(track.pre, track.samples[frame]);
        Matrix pose{};
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c)
                pose[r * 4 + c] = disk[c * 4 + r];

        const Factors factors = factor(pose, label + QString(" sample %1").arg(frame));
        const double sourceTime = double(frame) / track.rate;
        if (!std::isfinite(sourceTime))
            fail(label + " sample time is not finite");

        if (sourceTime > duration)
            continue;

        const float time = real(sourceTime, label + " sample time");
        if (!times.isEmpty() && time <= previous)
            fail(label + " sample times collide at float32 precision");

        times.append(time);
        append(translations, factors.translation, label);
        append(scales, factors.scale, label);
        appendRotation(left, factors.left, label);
        appendRotation(right, factors.right, label);
        last = factors;
        haveLast = true;
        previous = time;
    }

    if (!haveLast)
        fail(label + " contains no sample at clip start");

    const float end = real(duration, label + " duration");
    if (end > previous) {
        times.append(end);
        append(translations, last.translation, label);
        append(scales, last.scale, label);
        appendRotation(left, last.left, label);
        appendRotation(right, last.right, label);
    }

    const int input = builder.input(times);
    builder.channel(input, translations, "VEC3", joint.helper, "translation", "STEP");
    builder.channel(input, scales, "VEC3", joint.helper, "scale", "STEP");
    builder.channel(input, left, "VEC4", joint.helper, "rotation", "STEP");
    builder.channel(input, right, "VEC4", joint.joint, "rotation", "STEP");
}
} // namespace

Rig writeAnimation(Document &document, const Scene &scene) {
    if (!scene.animation)
        return {};

    const auto &source = scene.animation->sourceData();
    const qsizetype count = source.sampled ? source.tracks.size() : source.bones.size();
    if (count > std::numeric_limits<int>::max() - 1)
        fail("Source joint palette is too large");

    QVector<Joint> joints;
    joints.reserve(count);
    QVector<float> inverseBinds;
    inverseBinds.reserve((count + 1) * 16);
    if (source.sampled) {
        for (qsizetype i = 0; i < count; ++i) {
            if (source.clips.isEmpty()) {
                const auto &track = source.tracks[i];
                const QString label = QString("Legacy matrix track %1").arg(i);
                if (!std::isfinite(track.rate) || track.rate <= 0 || track.samples.isEmpty())
                    fail(label + " has an invalid sample rate or no samples");

                diskAffine(track.pre, label + " pre-matrix");
                for (const Matrix &sample : track.samples) {
                    diskAffine(sample, label + " disk sample");
                    const Matrix disk = multiply(track.pre, sample);
                    Matrix pose{};
                    for (int r = 0; r < 4; ++r)
                        for (int c = 0; c < 4; ++c)
                            pose[r * 4 + c] = disk[c * 4 + r];

                    factor(pose, label);
                }
            }

            const Joint joint = addJoint(document, QString("Legacy matrix track %1").arg(i), Factors{});
            joints.append(joint);
            document.roots.append(joint.helper);
            appendMatrix(inverseBinds, model::identity(), "Legacy identity inverse bind");
        }
    } else {
        QVector<QVector<int>> children(count);
        QVector<int> order;
        for (qsizetype i = 0; i < count; ++i) {
            const auto &bone = source.bones[i];
            const QString label = QString("Bone %1 (%2)").arg(i).arg(bone.name);
            if (bone.parent < -1 || bone.parent >= count)
                fail(label + " has an invalid parent index");

            if (bone.parent == -1)
                order.append(int(i));
            else
                children[bone.parent].append(int(i));

            joints.append(addJoint(document, bone.name, factor(bone.rest, label + " rest")));
        }

        for (qsizetype cursor = 0; cursor < order.size(); ++cursor)
            for (int child : children[order[cursor]])
                order.append(child);

        if (order.size() != count)
            fail("Skeleton parent hierarchy contains a cycle");

        QVector<Matrix> global(count), inverseBind(count);
        for (int i : order) {
            const auto &bone = source.bones[i];
            global[i] = bone.parent == -1 ? bone.rest : multiply(global[bone.parent], bone.rest);
            const QString label = QString("Bone %1 cumulative rest").arg(i);
            affine(global[i], label);
            inverseBind[i] = inverse(global[i], label);
            if (bone.parent == -1)
                document.roots.append(joints[i].helper);
            else
                document.child(joints[bone.parent].joint, joints[i].helper);
        }

        for (qsizetype i = 0; i < count; ++i)
            appendMatrix(inverseBinds, inverseBind[i], QString("Bone %1 inverse bind").arg(i));
    }

    const int identityNode = int(document.nodes.size());
    document.nodes.append(QJsonObject{{"name", "Unweighted identity joint"}});
    document.roots.append(identityNode);
    appendMatrix(inverseBinds, model::identity(), "Identity inverse bind");
    QJsonArray palette;
    for (const auto &joint : joints)
        palette.append(joint.joint);

    palette.append(identityNode);
    const int bindAccessor = document.floats(
        std::span<const float>(inverseBinds.constData(), size_t(inverseBinds.size())), "MAT4");
    Rig result{int(document.skins.size()), int(count)};
    document.skins.append(QJsonObject{{"name", scene.name + " skin"}, {"joints", palette},
                                      {"inverseBindMatrices", bindAccessor}});
    bool discontinuities = false;
    for (const auto &clip : source.clips) {
        const QString label = "Clip " + clip.name;
        real(clip.duration, label + " duration");
        if (clip.duration < 0)
            fail(label + " duration must be nonnegative");

        if (clip.duration > 0 && float(clip.duration) == 0)
            fail(label + " duration underflows float32");

        if (!std::isfinite(clip.timeParameter))
            fail(label + " timeParameter must be finite");

        QVector<bool> seen(count, false);
        for (const auto &channel : clip.channels) {
            if (channel.bone < 0 || channel.bone >= count)
                fail(label + " channel refers to a missing joint");

            if (seen[channel.bone])
                fail(label + " has duplicate channels for a joint");

            seen[channel.bone] = true;
        }

        AnimationBuilder builder{document, {}, {}};
        if (source.sampled) {
            if (!clip.channels.isEmpty())
                fail(label + " unexpectedly mixes sampled matrix tracks and skeleton channels");

            for (qsizetype i = 0; i < count; ++i)
                sampledTrack(builder, source.tracks[i], joints[i], clip.duration,
                             QString("Legacy matrix track %1").arg(i));
        } else {
            for (const auto &channel : clip.channels)
                modernChannel(builder, channel, joints[channel.bone], clip.duration,
                              label + QString(", bone %1").arg(channel.bone), discontinuities);
        }

        builder.finish(clip, identityNode);
    }

    if (discontinuities)
        document.warnings.append("Animation hold/scale discontinuities are encoded between adjacent float32 times; smooth intervals retain linear/spherical interpolation.");

    return result;
}
} // namespace ds2::glb
