#pragma once

#include "scene.h"
#include <array>

namespace ds2::model {
  using Matrix = std::array<double, 16>; // row-major, column-vector unless explicitly disk order
  Matrix identity();

  struct Bone {
    QString name;
    int parent = -1;
    Matrix rest;
  };

  struct Key {
    uint8_t flags = 0;
    std::array<double, 4> rotation{}; // XYZW
    std::array<double, 3> translation{}, scale{};
    double time = 0;
  };

  struct Channel {
    int bone = 0;
    QVector<Key> keys;
  };

  struct Clip {
    QString name;
    double duration = 0, timeParameter = 0;
    QVector<Channel> channels;
  };

  struct Track {
    double rate = 0;
    Matrix pre  = identity(); // original disk row-vector order
    QVector<Matrix> samples;  // original disk row-vector order
  };

  struct AnimationData {
    bool sampled = false;
    QVector<Bone> bones;
    QVector<Clip> clips;
    QVector<Track> tracks;
  };

  // Validates hierarchy/bind matrices and metadata, caches conservative bounds,
  // and starts the first clip when present. Scene geometry is not retained.
  std::shared_ptr<Animation> makeAnimation(AnimationData data, const Scene& scene);
} // namespace ds2::model
