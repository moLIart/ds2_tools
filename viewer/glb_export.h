#pragma once

#include "scene.h"
#include <QByteArray>
#include <QJsonArray>
#include <QJsonObject>
#include <span>

namespace ds2 {
  // Exports the stored model and every clip, independently of playback state.
  // Throws std::runtime_error on failure; commits the destination atomically.
  QStringList exportGlb(const Scene& scene, const QString& path);

  namespace glb {
    // Shared implementation surface for the container and animation writers.
    class Document {
    public:
      QByteArray binary;
      QJsonArray nodes, roots, meshes, materials, textures, images, samplers;
      QJsonArray skins, animations, accessors, bufferViews;
      QStringList warnings;
      int floats(std::span<const float> values, const char* type, bool bounds = false, int target = 0);
      int unsignedShorts(std::span<const quint16> values, const char* type, int target = 0);
      int unsignedInts(std::span<const quint32> values, const char* type, int target = 0);
      int bytes(const QByteArray& values, int target = 0);
      void child(int parent, int node);
    };

    struct Rig {
      // Palette order matches Scene::influences. The last joint is identity,
      // used for unweighted vertices in otherwise skinned meshes.
      int skin          = -1;
      int identityJoint = -1;
    };

    Rig writeAnimation(Document& document, const Scene& scene);
  } // namespace glb
} // namespace ds2
