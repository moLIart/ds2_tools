#include "material.h"

#include <cmath>
#include <memory>
#include <mutex>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QRegularExpression>
#include <stdexcept>
#include <utility>

namespace ds2 {
  namespace {
    [[noreturn]] void fail(const QString& message) {
      throw std::runtime_error(message.toUtf8().constData());
    }

    QString decode1251(const QByteArray& bytes) {
      // QString's built-in UTF codecs do not include Windows-1251.
      static constexpr char16_t upper[] = {0x0402, 0x0403, 0x201a, 0x0453, 0x201e, 0x2026, 0x2020, 0x2021, 0x20ac, 0x2030, 0x0409, 0x2039, 0x040a, 0x040c, 0x040b, 0x040f,
                                           0x0452, 0x2018, 0x2019, 0x201c, 0x201d, 0x2022, 0x2013, 0x2014, 0xfffd, 0x2122, 0x0459, 0x203a, 0x045a, 0x045c, 0x045b, 0x045f,
                                           0x00a0, 0x040e, 0x045e, 0x0408, 0x00a4, 0x0490, 0x00a6, 0x00a7, 0x0401, 0x00a9, 0x0404, 0x00ab, 0x00ac, 0x00ad, 0x00ae, 0x0407,
                                           0x00b0, 0x00b1, 0x0406, 0x0456, 0x0491, 0x00b5, 0x00b6, 0x00b7, 0x0451, 0x2116, 0x0454, 0x00bb, 0x0458, 0x0405, 0x0455, 0x0457};
      QString text;
      text.resize(bytes.size());
      for (qsizetype i = 0; i < bytes.size(); ++i) {
        const auto c = static_cast<unsigned char>(bytes[i]);
        text[i]      = QChar(c < 128 ? char16_t(c) : c < 192 ? upper[c - 128] : char16_t(0x0410 + c - 192));
      }

      return text;
    }

    using Fields = QHash<QString, QStringList>;

    Fields fields(const QStringList& tokens, qsizetype start, qsizetype end) {
      Fields result;
      qsizetype i = start;
      while (i < end) {
        const QString name = tokens[i++].toCaseFolded();
        if (i == end) {
          if (name == "diffuse_texture_transform") {
            result.insert(name, QStringList{});
          }

          break;
        }

        const QString value = tokens[i++];
        QStringList group;
        if (value == "{") {
          int depth = 1;
          while (i < end && depth) {
            const QString token = tokens[i++];
            depth += (token == "{") - (token == "}");
            if (depth) {
              group.append(token);
            }
          }
        } else {
          group.append(value);
        }

        result.insert(name, group);
      }

      return result;
    }

    QString unquote(QString text) {
      while (text.startsWith('"')) {
        text.remove(0, 1);
      }

      while (text.endsWith('"')) {
        text.chop(1);
      }

      return text;
    }

    float materialFloat(const QString& text, const QString& path) {
      bool ok           = false;
      const float value = text.toFloat(&ok);
      if (!ok || !std::isfinite(value)) {
        fail(QString("Invalid material number '%1' in %2").arg(text, path));
      }

      return value;
    }

    TextureTransform textureTransform(const QStringList& tokens, const QString& context) {
      if (tokens.isEmpty()) {
        fail(QString("Empty diffuse_texture_transform in %1").arg(context));
      }

      TextureTransform transform;
      const QString operation = unquote(tokens.first()).toCaseFolded();
      if (operation == "scroll") {
        if (tokens.size() < 3) {
          fail(QString("Incomplete diffuse_texture_transform scroll in %1: expected two rates").arg(context));
        }

        transform.type = TextureTransform::Type::Scroll;
        transform.rate = {materialFloat(tokens[1], context), materialFloat(tokens[2], context)};
      } else if (operation == "rotate") {
        if (tokens.size() < 2) {
          fail(QString("Incomplete diffuse_texture_transform rotate in %1: expected one rate").arg(context));
        }

        transform.type = TextureTransform::Type::Rotate;
        transform.rate.setX(materialFloat(tokens[1], context));
        // The engine reads only the first rate, even when declarations contain more.
      } else {
        fail(QString("Unsupported diffuse_texture_transform operation '%1' in %2").arg(operation, context));
      }

      return transform;
    }
  } // namespace

  QHash<QString, MaterialDeclaration> readMaterials(const QString& directory) {
    QHash<QString, MaterialDeclaration> result;
    const auto paths = QDir(directory).entryInfoList({"*.mat"}, QDir::Files, QDir::Name | QDir::IgnoreCase);
    const QRegularExpression comments("/\\*.*?(?:\\*/|\\z)|//[^\\r\\n]*", QRegularExpression::DotMatchesEverythingOption);
    const QRegularExpression tokenizer("\"[^\"]*\"|[{}]|[^\\s{}]+");
    for (const auto& path : paths) {
      QFile file(path.absoluteFilePath());
      if (!file.open(QIODevice::ReadOnly)) {
        fail(QString("Cannot read material %1: %2").arg(path.absoluteFilePath(), file.errorString()));
      }

      const auto bytes = file.readAll();
      if (file.error() != QFileDevice::NoError) {
        fail(QString("Cannot read material %1: %2").arg(path.absoluteFilePath(), file.errorString()));
      }

      QString text = decode1251(bytes);
      text.remove(comments);
      QStringList tokens;
      auto matches = tokenizer.globalMatch(text);
      while (matches.hasNext()) {
        tokens.append(matches.next().captured());
      }

      qsizetype i = 0;
      while (i + 2 < tokens.size()) {
        if (tokens[i].toCaseFolded() != "template_material" || tokens[i + 2] != "{") {
          ++i;
          continue;
        }

        MaterialDeclaration d;
        d.name = tokens[i + 1];
        const QString context = QString("material '%1' in %2").arg(d.name, path.absoluteFilePath());
        i += 3;
        const auto start = i;
        int depth        = 1;
        bool hasTransform = false;
        while (i < tokens.size() && depth) {
          const auto& token = tokens[i++];
          depth += (token == "{") - (token == "}");
          hasTransform |= token.compare("diffuse_texture_transform", Qt::CaseInsensitive) == 0;
        }

        if (depth && hasTransform) {
          fail(QString("Unclosed diffuse_texture_transform declaration in %1").arg(context));
        }

        const auto f      = fields(tokens, start, i - 1);
        const auto scalar = [&](const QString& key, const QString& fallback = QString()) {
          const auto it = f.constFind(key);
          if (it == f.cend()) {
            return fallback;
          }

          if (it->isEmpty()) {
            fail(QString("Empty material field '%1' in %2").arg(key, path.absoluteFilePath()));
          }

          return unquote(it->first()).toCaseFolded();
        };
        d.shader           = scalar("material", "unknown");
        d.vertexColors     = scalar("use_vertex_colors") == "true" || scalar("color") == "vertex";
        d.selfIllumination = scalar("self_illumination") == "true";
        d.overbright       = scalar("overbright", "true") != "false";
        d.blend            = scalar("blend");
        d.cull             = scalar("cull_face") != "disable";
        d.depthWrite       = scalar("depth_write") != "false" && scalar("depth_write") != "disable";
        d.depthTest        = scalar("depth_test") != "disable";
        d.depthOffset      = scalar("depth_offset") == "true";
        bool sortOk        = false;
        d.sort             = scalar("sort_value", "5").toInt(&sortOk);
        if (!sortOk) {
          fail(QString("Invalid sort_value in %1").arg(path.absoluteFilePath()));
        }

        const auto alpha = f.value("alphatest");
        if (alpha.size() >= 2 && alpha[0] == "greater") {
          d.alphaCutoff = materialFloat(alpha[1], path.absoluteFilePath());
        }

        const auto color = f.value("color3", {"1", "1", "1"});
        if (color.size() < 3) {
          fail(QString("Incomplete color3 in %1").arg(path.absoluteFilePath()));
        }

        d.color = {materialFloat(color[0], path.absoluteFilePath()), materialFloat(color[1], path.absoluteFilePath()), materialFloat(color[2], path.absoluteFilePath())};
        const auto diffuse = f.value("diffuse_texture", {"$white$"});
        if (diffuse.isEmpty()) {
          fail(QString("Empty diffuse_texture in %1").arg(path.absoluteFilePath()));
        }

        d.diffuse = unquote(diffuse[0]);
        d.clamp   = diffuse.contains("clampmap");
        const auto transform = f.constFind("diffuse_texture_transform");
        if (transform != f.cend()) {
          d.diffuseTransform = textureTransform(*transform, context);
        }

        result.insert(d.name.toCaseFolded(), std::move(d));
      }
    }

    return result;
  }

  const QHash<QString, MaterialDeclaration>& cachedMaterials(const QString& directory) {
    QString key = QFileInfo(directory).canonicalFilePath();
    if (key.isEmpty()) {
      key = QDir(directory).absolutePath();
    }

    key = QDir::cleanPath(QDir::fromNativeSeparators(key));
#ifdef Q_OS_WIN
    key = key.toCaseFolded();
#endif
    static std::mutex mutex;
    static QHash<QString, std::shared_ptr<const QHash<QString, MaterialDeclaration>>> declarations;
    const std::lock_guard lock(mutex);
    auto it = declarations.constFind(key);
    if (it == declarations.cend()) {
      it = declarations.insert(key, std::make_shared<const QHash<QString, MaterialDeclaration>>(readMaterials(directory)));
    }

    return **it;
  }
} // namespace ds2
