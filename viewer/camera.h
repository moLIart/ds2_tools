#pragma once
#include "scene.h"
#include <QSet>
#include <QtMath>
#include <algorithm>
#include <cmath>

namespace ds2 {
class Camera {
  public:
    bool map = false, fly = false;
    QVector3D target, flyEye;
    float yaw = -.95f, pitch = .45f, distance = 10, radius = 1, speed = 200;
    float drawDistance = 0; // zero: automatic
    float lightYaw = 0, lightPitch = 0;
    Camera(bool isMap = false) : map(isMap) {
        resetLight();
    }
    QVector3D forward() const {
        return {-std::cos(pitch) * std::cos(yaw), -std::cos(pitch) * std::sin(yaw), -std::sin(pitch)};
    }
    QVector3D eye() const {
        return fly ? flyEye : target - forward() * distance;
    }
    QVector3D light() const {
        return {std::cos(lightPitch) * std::cos(lightYaw), std::cos(lightPitch) * std::sin(lightYaw),
                std::sin(lightPitch)};
    }
    void resetLight() {
        lightYaw = std::atan2(.55f, -.35f);
        lightPitch = std::atan2(.85f, std::hypot(.35f, -.55f));
    }
    void fit(const Bounds &bounds, float aspect) {
        target = (bounds.low + bounds.high) * .5f;
        radius = std::max((bounds.high - bounds.low).length() * .5f, map ? 1.f : .001f);
        const float half = qDegreesToRadians(map ? 25.f : 22.5f);
        const float angle = std::min(half, std::atan(std::tan(half) * aspect));
        distance = radius / std::sin(angle) * (map ? 1.1f : 1.25f);
        yaw = map ? -.95f : qDegreesToRadians(125.f);
        pitch = map ? .45f : qDegreesToRadians(20.f);
        fly = false;
        speed = std::max(10.f, radius * .1f);
    }
    void setFly(bool enabled) {
        if (enabled && !fly)
            flyEye = eye();
        else if (fly && !enabled)
            target = flyEye + forward() * distance;
        fly = enabled;
    }
    void drag(float dx, float dy, Qt::MouseButtons buttons, Qt::KeyboardModifiers mods, int height) {
        // Qt Y deltas point down; Python's pointer deltas pointed up.
        dy = -dy;
        if (map) {
            if (buttons & (Qt::LeftButton | Qt::RightButton)) {
                const float sign = fly ? 1.f : -1.f;
                yaw -= sign * dx * .005f;
                pitch = std::clamp(pitch - sign * dy * .005f, -1.55f, 1.55f);
            } else if (buttons & Qt::MiddleButton) {
                const QVector3D right(-std::sin(yaw), std::cos(yaw), 0);
                const auto d = (right * (-dx) + QVector3D(0, 0, -dy)) * distance / std::max(height, 1);
                if (fly)
                    flyEye += d;
                else
                    target += d;
            }
        } else if (buttons & (Qt::RightButton | Qt::MiddleButton)) {
            const auto f = forward(), right = QVector3D::crossProduct(f, {0, 0, 1}).normalized(),
                       up = QVector3D::crossProduct(right, f).normalized();
            target += (right * (-dx) + up * (-dy)) *
                      (2 * distance * std::tan(qDegreesToRadians(22.5f)) / std::max(height, 1));
        } else if (buttons & Qt::LeftButton) {
            float &y = (mods & Qt::ShiftModifier) ? lightYaw : yaw,
                  &p = (mods & Qt::ShiftModifier) ? lightPitch : pitch;
            y -= dx * .008f;
            p = std::clamp(p - dy * .008f, qDegreesToRadians(-89.f), qDegreesToRadians(89.f));
        }
    }
    void wheel(float steps) {
        if (map && fly)
            speed = std::clamp(speed * std::exp(steps * .15f), .1f, 1e8f);
        else if (map)
            distance = std::clamp(distance * std::exp(-steps * .15f), .05f, radius * 100);
        else
            distance = std::clamp(distance * std::exp(std::clamp(-steps * .12f, -20.f, 20.f)), radius * .02f,
                                  radius * 1000);
    }
    bool move(const QSet<int> &keys, float dt) {
        if (!map || !fly)
            return false;
        const auto f = forward(), right = QVector3D::crossProduct(f, {0, 0, 1}).normalized();
        const auto m = f * float(keys.contains(Qt::Key_W) - keys.contains(Qt::Key_S)) +
                       right * float(keys.contains(Qt::Key_D) - keys.contains(Qt::Key_A)) +
                       QVector3D(0, 0, keys.contains(Qt::Key_E) - keys.contains(Qt::Key_Q));
        if (m.lengthSquared() == 0)
            return false;
        flyEye += m.normalized() * speed * std::min(dt, .1f) * (keys.contains(Qt::Key_Shift) ? 5.f : 1.f);
        return true;
    }
    std::pair<float, float> planes(const Bounds &bounds, bool cap = true) const {
        const auto f = forward(), e = eye(), center = (bounds.low + bounds.high) * .5f,
                   half = (bounds.high - bounds.low) * .5f;
        const float cd = QVector3D::dotProduct(center - e, f),
                    hd = half.x() * std::abs(f.x()) + half.y() * std::abs(f.y()) + half.z() * std::abs(f.z());
        const float pad = map ? std::max(1.f, hd * .01f) : std::max(hd * .1f, distance * .001f);
        float near = std::max(map ? .5f : distance * .01f, cd - hd - pad),
              far = std::max(near + (map ? 1.f : pad), cd + hd + pad);
        if (map && cap && drawDistance > 0) {
            far = std::min(far, drawDistance);
            near = std::min(near, far - .5f);
        }
        return {near, far};
    }
    QMatrix4x4 clip(const Bounds &bounds, float aspect, bool zeroToOne = false) const {
        const auto [n, f] = planes(bounds);
        QMatrix4x4 projection;
        if (map) {
            projection.fill(0);
            const float s = 1 / std::tan(qDegreesToRadians(25.f)), d = n / (f - n);
            projection(0, 0) = s / aspect;
            projection(1, 1) = s;
            projection(2, 2) = zeroToOne ? d : 2 * d + 1;
            projection(2, 3) = zeroToOne ? f * d : 2 * f * d;
            projection(3, 2) = -1;
        } else
            projection.perspective(45.f, aspect, n, f);
        QMatrix4x4 view;
        view.lookAt(eye(), eye() + forward(), {0, 0, 1});
        return projection * view;
    }
};
} // namespace ds2
