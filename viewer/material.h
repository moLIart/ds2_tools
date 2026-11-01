#pragma once

#include "scene.h"

#include <QHash>

namespace ds2 {
  struct MaterialDeclaration {
    QString name, shader = "unknown", diffuse = "$diffuse$", blend;
    QVector3D color{1, 1, 1};
    TextureTransform diffuseTransform;
    bool vertexColors = false, selfIllumination = false, overbright = true;
    bool cull = true, depthWrite = true, depthTest = true, depthOffset = false;
    bool clamp = false;
    float alphaCutoff = -1;
    int sort          = 5;
  };

  QHash<QString, MaterialDeclaration> readMaterials(const QString& directory);
  const QHash<QString, MaterialDeclaration>& cachedMaterials(const QString& directory);
} // namespace ds2
