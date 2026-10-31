#include "render_widget.h"
#include "texture.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <numeric>
#include <QApplication>
#include <QCursor>
#include <QDebug>
#include <QDragEnterEvent>
#include <QFile>
#include <QFocusEvent>
#include <QKeyEvent>
#include <QMimeData>
#include <QMouseEvent>
#include <QQuaternion>
#include <QTimer>
#include <QWheelEvent>
#include <rhi/qrhi.h>
#include <stdexcept>
#ifdef Q_OS_WIN
  #define WIN32_LEAN_AND_MEAN
  #define NOMINMAX
  #include <windows.h>
#endif

namespace ds2 {
  namespace {
    void require(bool ok, const char* message) {
      if (!ok) {
        throw std::runtime_error(message);
      }
    }

    QShader shader(const char* path) {
      QFile f(QString::fromLatin1(path));
      require(f.open(QIODevice::ReadOnly), "Cannot open shader resource");
      auto s = QShader::fromSerialized(f.readAll());
      require(s.isValid(), "Invalid shader resource");
      return s;
    }

    struct FrameUniform {
      float mvp[16];
      float light[4];
      int params[4];
    };

    struct DrawUniform {
      float model[16];
      float tintGain[4];
      float meshCutoff[4];
      int flags[4];
    };

    static_assert(sizeof(FrameUniform) == 96 && sizeof(DrawUniform) == 112);

    QSize dataSize(qsizetype count) {
      const int w = int(std::min<qsizetype>(std::max<qsizetype>(1, count), 1024));
      return {w, int((std::max<qsizetype>(1, count) + w - 1) / w)};
    }

    Material lightMaterial() {
      Material m;
      m.albedo     = "$white$";
      m.blend      = Blend::Alpha;
      m.cull       = true;
      m.depthWrite = false;
      return m;
    }

    void appendSphere(QVector<Vertex>& vertices, QVector<quint32>& indices) {
      constexpr int segments = 16, rings = 8;
      constexpr float pi     = 3.14159265358979323846f;
      const quint32 base     = quint32(vertices.size());
      vertices.reserve(vertices.size() + (rings + 1) * (segments + 1));
      indices.reserve(indices.size() + rings * segments * 6);
      for (int lat = 0; lat <= rings; ++lat) {
        const float theta = pi * lat / rings;
        const float z = std::cos(theta), radius = std::sin(theta);
        for (int lon = 0; lon <= segments; ++lon) {
          const float phi = 2 * pi * lon / segments;
          Vertex v;
          v.position[0] = radius * std::cos(phi);
          v.position[1] = radius * std::sin(phi);
          v.position[2] = z;
          vertices.append(v);
        }
      }

      for (int lat = 0; lat < rings; ++lat) {
        for (int lon = 0; lon < segments; ++lon) {
          const quint32 a = base + lat * (segments + 1) + lon, b = a + segments + 1;
          indices.append(a);
          indices.append(b);
          indices.append(a + 1);
          indices.append(a + 1);
          indices.append(b);
          indices.append(b + 1);
        }
      }
    }

    void appendCone(QVector<Vertex>& vertices, QVector<quint32>& indices) {
      constexpr int segments = 16;
      constexpr float pi = 3.14159265358979323846f, radius = .3f, height = 1.f;
      const quint32 apex = 0;
      Vertex v;
      vertices.append(v);
      for (int i = 0; i < segments; ++i) {
        const float angle = 2 * pi * i / segments;
        v.position[0] = radius * std::cos(angle);
        v.position[1] = radius * std::sin(angle);
        v.position[2] = height;
        vertices.append(v);
      }

      const quint32 center = segments + 1;
      v.position[0] = v.position[1] = 0;
      vertices.append(v);
      for (int i = 0; i < segments; ++i) {
        const quint32 current = 1 + i, next = 1 + (i + 1) % segments;
        indices.append(apex);
        indices.append(next);
        indices.append(current);
        indices.append(center);
        indices.append(next);
        indices.append(current);
      }
    }
  } // namespace

  struct RenderWidget::Target {
    std::unique_ptr<QRhiTexture> depth;
    std::unique_ptr<QRhiRenderPassDescriptor> pass;
    std::unique_ptr<QRhiTextureRenderTarget> rt;
  };

  struct RenderWidget::Data {
    std::shared_ptr<Scene> scene;
    std::unique_ptr<QRhiBuffer> vertices, indices, frame, draw;
    std::unique_ptr<QRhiTexture> influences, palette;
    std::unique_ptr<QRhiSampler> repeat, clamp, nearest;
    std::map<QString, UploadedTexture> textures;
    std::map<QString, std::unique_ptr<QRhiShaderResourceBindings>> bindings;
    std::map<uint32_t, std::unique_ptr<QRhiGraphicsPipeline>> pipelines;

    struct MeshGpu {
      quint32 vertexOffset, indexStart, indexCount;
      QVector3D center;
      QRhiShaderResourceBindings* bindings;
    };

    struct LightGpu {
      quint32 vertexOffset, indexStart, indexCount;
    };

    QVector<MeshGpu> meshes;
    QVector<int> order;
    QVector<float> sortDistances;
    QVector<LightGpu> lights;
    QVector<int> lightOrder;
    QVector<float> lightDistances;
    QVector<float> paletteBytes;
    int stride     = 0;
    bool unblended = false;
    QSet<QString> missingTextures;
    quint32 gridVertexOffset = 0, gridVertexCount = 0;
    quint32 crossVertexOffset = 0, crossVertexCount = 0;
    QRhiShaderResourceBindings* gridBindings = nullptr;
    QRhiShaderResourceBindings* lightBindings = nullptr;
    Bounds gridBounds;
    QShader vert, frag;

    QRhiTexture* texture(QRhi* rhi, QRhiResourceUpdateBatch* batch, const QString& path) {
      const QString fallback = scene->map ? "$missing$" : "$white$";
      if (missingTextures.contains(path)) {
        return textures.at(fallback).texture.get();
      }

      if (auto it = textures.find(path); it != textures.end()) {
        return it->second.texture.get();
      }

      try {
        auto t  = loadTexture(rhi, batch, path);
        auto* p = t.texture.get();
        textures.emplace(path, std::move(t));
        return p;
      } catch (const std::exception& e) {
        if (path.startsWith('$')) {
          throw;
        }

        scene->warnings.append(QString("%1: %2").arg(path, QString::fromUtf8(e.what())));
        auto* result = texture(rhi, batch, fallback);
        missingTextures.insert(path);
        return result;
      }
    }

    int lastMip(const QString& path) const {
      const QString key = missingTextures.contains(path) ? (scene->map ? "$missing$" : "$white$") : path;
      return textures.at(key).lastMip;
    }

    QRhiShaderResourceBindings* binding(QRhi* rhi, QRhiResourceUpdateBatch* batch, const Material& m) {
      const auto key = m.albedo + '\n' + m.lightmap + (m.clamp ? "\nC" : "\nR");
      if (auto it = bindings.find(key); it != bindings.end()) {
        return it->second.get();
      }

      auto s  = std::unique_ptr<QRhiShaderResourceBindings>(rhi->newShaderResourceBindings());
      using B = QRhiShaderResourceBinding;
      s->setBindings({B::uniformBuffer(0, B::VertexStage | B::FragmentStage, frame.get()),
                      B::uniformBufferWithDynamicOffset(1, B::VertexStage | B::FragmentStage, draw.get(), sizeof(DrawUniform)),
                      B::sampledTexture(2, B::FragmentStage, texture(rhi, batch, m.albedo), m.clamp ? clamp.get() : repeat.get()),
                      B::sampledTexture(3, B::FragmentStage, texture(rhi, batch, m.lightmap), clamp.get()), B::sampledTexture(4, B::VertexStage, influences.get(), nearest.get()),
                      B::sampledTexture(5, B::VertexStage, palette.get(), nearest.get())});
      require(s->create(), "Cannot create shader bindings");
      auto* p = s.get();
      bindings.emplace(key, std::move(s));
      return p;
    }

    void uploadPalette(QRhiResourceUpdateBatch* batch) {
      if (scene->animation && !scene->animation->matrices().isEmpty()) {
        const auto& m = scene->animation->matrices();
        for (qsizetype i = 0; i < m.size(); ++i) {
          std::memcpy(paletteBytes.data() + i * 16, m[i].constData(), 64);
        }
      } else {
        QMatrix4x4 m;
        std::memcpy(paletteBytes.data(), m.constData(), 64);
      }

      batch->uploadTexture(palette.get(), {{0, 0, QRhiTextureSubresourceUploadDescription(paletteBytes.constData(), quint32(paletteBytes.size() * sizeof(float)))}});
    }

    uint32_t pipelineKey(const Material& m, bool wire, bool solid, bool grid, bool light = false) const {
      return uint32_t(m.cull) | uint32_t(m.depthTest || solid) << 1 | uint32_t(m.depthWrite || solid) << 2 | uint32_t(m.depthOffset) << 3 |
             uint32_t(solid || (unblended && !light) ? Blend::None : m.blend) << 4 | uint32_t(wire && !grid) << 8 | uint32_t(grid) << 9 | uint32_t(light) << 10;
    }

    QRhiGraphicsPipeline* pipeline(QRhi* rhi, QRhiRenderPassDescriptor* pass, const Material& m, bool wire, bool solid, bool grid, int samples, bool light = false) {
      const auto key = pipelineKey(m, wire, solid, grid, light);
      if (auto it = pipelines.find(key); it != pipelines.end()) {
        return it->second.get();
      }

      auto p = std::unique_ptr<QRhiGraphicsPipeline>(rhi->newGraphicsPipeline());
      p->setShaderStages({{QRhiShaderStage::Vertex, vert}, {QRhiShaderStage::Fragment, frag}});
      QRhiVertexInputLayout layout;
      layout.setBindings({{sizeof(Vertex)}});
      layout.setAttributes({{0, 0, QRhiVertexInputAttribute::Float3, 0},
                            {0, 1, QRhiVertexInputAttribute::Float3, 12},
                            {0, 2, QRhiVertexInputAttribute::Float2, 24},
                            {0, 3, QRhiVertexInputAttribute::Float2, 32},
                            {0, 4, QRhiVertexInputAttribute::Float4, 40},
                            {0, 5, QRhiVertexInputAttribute::UInt2, 56}});
      p->setVertexInputLayout(layout);
      p->setShaderResourceBindings(meshes.isEmpty() ? gridBindings : meshes.first().bindings);
      p->setRenderPassDescriptor(pass);
      p->setSampleCount(samples);
      p->setTopology(grid ? QRhiGraphicsPipeline::Lines : QRhiGraphicsPipeline::Triangles);
      p->setCullMode(m.cull ? QRhiGraphicsPipeline::Back : QRhiGraphicsPipeline::None);
      p->setFrontFace(QRhiGraphicsPipeline::CCW);
      p->setDepthTest(m.depthTest || solid);
      p->setDepthWrite(m.depthWrite || solid);
      p->setDepthOp(scene->map ? QRhiGraphicsPipeline::Greater : QRhiGraphicsPipeline::LessOrEqual);
      if (m.depthOffset) {
        p->setDepthBias(1);
        p->setSlopeScaledDepthBias(1);
      }

      if (wire && !grid) {
        require(rhi->isFeatureSupported(QRhi::NonFillPolygonMode), "Backend does not support wireframe");
        p->setPolygonMode(QRhiGraphicsPipeline::Line);
      }

      QRhiGraphicsPipeline::TargetBlend blend;
      blend.enable = !solid && (light || !unblended) && m.blend != Blend::None;
      if (m.blend == Blend::Alpha) {
        blend.srcColor = blend.srcAlpha = QRhiGraphicsPipeline::SrcAlpha;
        blend.dstColor = blend.dstAlpha = QRhiGraphicsPipeline::OneMinusSrcAlpha;
      } else if (m.blend == Blend::Add) {
        blend.srcColor = blend.srcAlpha = QRhiGraphicsPipeline::One;
        blend.dstColor = blend.dstAlpha = QRhiGraphicsPipeline::One;
      } else if (m.blend == Blend::Modulate) {
        blend.srcColor = QRhiGraphicsPipeline::DstColor;
        blend.srcAlpha = QRhiGraphicsPipeline::DstAlpha;
        blend.dstColor = blend.dstAlpha = QRhiGraphicsPipeline::Zero;
      } else {
        blend.srcColor = blend.srcAlpha = QRhiGraphicsPipeline::One;
        blend.dstColor = blend.dstAlpha = QRhiGraphicsPipeline::Zero;
      }

      p->setTargetBlends({blend});
      require(p->create(), "Cannot create graphics pipeline");
      auto* result = p.get();
      pipelines.emplace(key, std::move(p));
      return result;
    }
  };

  RenderWidget::RenderWidget(bool map, QWidget* parent) : QRhiWidget(parent), camera(map), m_map(map) {
    setAutoRenderTarget(false);
    setSampleCount(map ? 4 : 1);
    setFocusPolicy(Qt::StrongFocus);
    setAcceptDrops(true);
    setMouseTracking(true);
#ifdef Q_OS_WIN
    // Keep Qt's grab window stable before the first mouse press.
    if (m_map) {
      setAttribute(Qt::WA_NativeWindow);
    }
#endif

    m_clock.start();
    m_fpsClock.start();
    auto* timer = new QTimer(this);
    timer->setInterval(map ? 0 : 16);
    connect(timer, &QTimer::timeout, this, [this] {
      tick();
    });
    timer->start();
    connect(this, &QRhiWidget::frameSubmitted, this, [this] {
      ++m_frames;
      if (m_fpsClock.elapsed() >= 500) {
        m_fps    = m_frames * 1000. / m_fpsClock.restart();
        m_frames = 0;
        if (changed) {
          changed();
        }
      }
    });
    connect(this, &QRhiWidget::renderFailed, this, [this] {
      if (failed) {
        failed("QRhi rendering failed. Check graphics driver/backend.");
      }
    });
  }

  RenderWidget::~RenderWidget() {
    releaseMouseCapture();
    releaseResources();
  }

  std::shared_ptr<Scene> RenderWidget::scene() const {
    return m_scene;
  }

  void RenderWidget::setScene(std::shared_ptr<Scene> scene) {
    m_pending = std::move(scene);
    update();
  }

  Bounds RenderWidget::bounds(bool grid) const {
    if (!m_scene) {
      return {};
    }

    auto b = m_poseBounds;
    if (grid && m_data && showGrid && !m_map) {
      for (int i = 0; i < 3; ++i) {
        b.low[i]  = std::min(b.low[i], m_data->gridBounds.low[i]);
        b.high[i] = std::max(b.high[i], m_data->gridBounds.high[i]);
      }
    }

    return b;
  }

  void RenderWidget::fit() {
    if (m_scene) {
      camera.fit(bounds(), float(width()) / std::max(height(), 1));
    }

    update();
    if (changed) {
      changed();
    }
  }

  void RenderWidget::poseChanged() {
    m_paletteDirty = true;
    if (m_scene) {
      m_poseBounds  = m_scene->animation ? m_scene->animation->bounds() : m_scene->bounds;
      camera.radius = std::max((m_poseBounds.high - m_poseBounds.low).length() * .5f, .001f);
    }

    update();
  }

  void RenderWidget::releaseResources() {
    m_data.reset();
    m_target.reset();
    m_owner = nullptr;
    if (m_scene) {
      m_pending = m_scene;
    }
  }

  void RenderWidget::initialize(QRhiCommandBuffer*) {
    try {
      if (m_owner != rhi()) {
        releaseResources();
        m_owner = rhi();
      }

      if (m_map) {
        require(rhi()->supportedSampleCounts().contains(4), "Map viewer requires 4x MSAA");
      }

      const QSize size = resolveTexture() ? resolveTexture()->pixelSize() : colorTexture()->pixelSize();
      if (!m_target || m_target->depth->pixelSize() != size) {
        if (m_data) {
          m_data->pipelines.clear();
        }

        m_target.reset();
        auto t = std::make_unique<Target>();
        require(rhi()->isTextureFormatSupported(QRhiTexture::D32F, QRhiTexture::RenderTarget), "D32F depth textures are unavailable");
        t->depth.reset(rhi()->newTexture(QRhiTexture::D32F, size, sampleCount(), QRhiTexture::RenderTarget));
        require(t->depth->create(), "Cannot create D32F depth texture");
        QRhiColorAttachment color = msaaColorBuffer() ? QRhiColorAttachment(msaaColorBuffer()) : QRhiColorAttachment(colorTexture());
        if (resolveTexture()) {
          color.setResolveTexture(resolveTexture());
        }

        QRhiTextureRenderTargetDescription desc;
        desc.setColorAttachments({color});
        desc.setDepthTexture(t->depth.get());
        t->rt.reset(rhi()->newTextureRenderTarget(desc));
        t->pass.reset(t->rt->newCompatibleRenderPassDescriptor());
        t->rt->setRenderPassDescriptor(t->pass.get());
        require(t->rt->create(), "Cannot create render target");
        m_target = std::move(t);
        qInfo() << "QRhi" << rhi()->backendName() << rhi()->driverInfo().deviceName << "D32F samples" << sampleCount();
      }
    } catch (const std::exception& e) {
      if (failed) {
        failed(QString::fromUtf8(e.what()));
      }
    }
  }

  void RenderWidget::render(QRhiCommandBuffer* cb) {
    if (!m_target) {
      return;
    }

    if (m_pending) {
      auto* batch = rhi()->nextResourceUpdateBatch();
      try {
        auto d    = std::make_unique<Data>();
        d->scene  = m_pending;
        d->vert   = shader(":/shaders/scene.vert.qsb");
        d->frag   = shader(":/shaders/scene.frag.qsb");
        d->stride = rhi()->ubufAligned(sizeof(DrawUniform));
        d->frame.reset(rhi()->newBuffer(QRhiBuffer::Dynamic, QRhiBuffer::UniformBuffer, sizeof(FrameUniform)));
        require(d->frame->create(), "Cannot create frame buffer");
        d->draw.reset(rhi()->newBuffer(QRhiBuffer::Dynamic, QRhiBuffer::UniformBuffer, d->stride * (d->scene->meshes.size() + d->scene->lights.size() + 1)));
        require(d->draw->create(), "Cannot create draw buffer");
        d->repeat.reset(rhi()->newSampler(QRhiSampler::Linear, QRhiSampler::Linear, QRhiSampler::Linear, QRhiSampler::Repeat, QRhiSampler::Repeat));
        d->clamp.reset(rhi()->newSampler(QRhiSampler::Linear, QRhiSampler::Linear, QRhiSampler::Linear, QRhiSampler::ClampToEdge, QRhiSampler::ClampToEdge));
        d->nearest.reset(rhi()->newSampler(QRhiSampler::Nearest, QRhiSampler::Nearest, QRhiSampler::None, QRhiSampler::ClampToEdge, QRhiSampler::ClampToEdge));
        require(d->repeat->create() && d->clamp->create() && d->nearest->create(), "Cannot create texture samplers");
        const auto influenceSize = dataSize(d->scene->influences.size());
        QVector<float> influences(influenceSize.width() * influenceSize.height() * 4, 0.f);
        for (qsizetype i = 0; i < d->scene->influences.size(); ++i) {
          influences[i * 4]     = d->scene->influences[i].x();
          influences[i * 4 + 1] = d->scene->influences[i].y();
        }

        d->influences.reset(rhi()->newTexture(QRhiTexture::RGBA32F, influenceSize));
        require(d->influences->create(), "Cannot create influence texture");
        batch->uploadTexture(d->influences.get(), {{0, 0, QRhiTextureSubresourceUploadDescription(influences.constData(), quint32(influences.size() * 4))}});
        const qsizetype matrices = d->scene->animation ? d->scene->animation->matrices().size() : 1;
        const auto paletteSize   = dataSize(std::max<qsizetype>(1, matrices) * 4);
        d->paletteBytes.resize(paletteSize.width() * paletteSize.height() * 4);
        d->palette.reset(rhi()->newTexture(QRhiTexture::RGBA32F, paletteSize));
        require(d->palette->create(), "Cannot create palette texture");
        d->uploadPalette(batch);
        QVector<quint32> offsets;
        qsizetype vertexBytes = 0, indexBytes = 0;
        for (const auto& g : d->scene->geometries) {
          offsets.append(quint32(vertexBytes));
          vertexBytes += g.vertices.size() * sizeof(Vertex);
        }

        for (const auto& m : d->scene->meshes) {
          indexBytes += m.indices.size() * 4;
        }

        QVector<Vertex> grid;
        if (!m_map) {
          const auto b       = d->scene->animation ? d->scene->animation->bounds() : d->scene->bounds;
          const auto center  = (b.low + b.high) * .5f;
          const float extent = std::max((b.high - b.low).length() * .5f, .001f) * 1.5f, z = b.low.z() - extent / 1.5f * .005f;
          d->gridBounds = {{center.x() - extent, center.y() - extent, z}, {center.x() + extent, center.y() + extent, z}};
          auto add      = [&](float x, float y) {
            Vertex v;
            v.position[0] = x;
            v.position[1] = y;
            v.position[2] = z;
            grid.append(v);
          };
          for (int i = -10; i <= 10; ++i) {
            const float p = i * extent / 10;
            add(center.x() - extent, center.y() + p);
            add(center.x() + extent, center.y() + p);
            add(center.x() + p, center.y() - extent);
            add(center.x() + p, center.y() + extent);
          }

          d->gridVertexOffset = quint32(vertexBytes);
          d->gridVertexCount  = grid.size();
          vertexBytes += grid.size() * sizeof(Vertex);
        }

        QVector<Vertex> lightVertices;
        QVector<quint32> lightIndices;
        quint32 lightVertexOffset = 0, lightIndexOffset = 0;
        quint32 sphereVertexOffset = 0, sphereIndexStart = 0, sphereIndexCount = 0;
        quint32 coneVertexOffset = 0, coneIndexStart = 0, coneIndexCount = 0;
        if (m_map && !d->scene->lights.isEmpty()) {
          lightVertexOffset = quint32(vertexBytes);
          lightIndexOffset  = quint32(indexBytes / sizeof(quint32));
          sphereVertexOffset = lightVertexOffset;
          sphereIndexStart   = lightIndexOffset;
          appendSphere(lightVertices, lightIndices);
          sphereIndexCount = quint32(lightIndices.size());
          coneVertexOffset  = lightVertexOffset + quint32(lightVertices.size() * sizeof(Vertex));
          coneIndexStart    = lightIndexOffset + quint32(lightIndices.size());
          appendCone(lightVertices, lightIndices);
          coneIndexCount = quint32(lightIndices.size()) - sphereIndexCount;
          d->crossVertexOffset = lightVertexOffset + quint32(lightVertices.size() * sizeof(Vertex));
          d->crossVertexCount  = 6;
          for (int axis = 0; axis < 3; ++axis) {
            for (const float sign : {-1.f, 1.f}) {
              Vertex v;
              v.position[axis] = sign * .1f;
              lightVertices.append(v);
            }
          }

          vertexBytes += lightVertices.size() * sizeof(Vertex);
          indexBytes += lightIndices.size() * sizeof(quint32);
        }

        require(vertexBytes > 0 && vertexBytes <= std::numeric_limits<quint32>::max() && indexBytes <= std::numeric_limits<quint32>::max(), "Scene exceeds GPU buffer limits");
        d->vertices.reset(rhi()->newBuffer(QRhiBuffer::Immutable, QRhiBuffer::VertexBuffer, quint32(vertexBytes)));
        d->indices.reset(rhi()->newBuffer(QRhiBuffer::Immutable, QRhiBuffer::IndexBuffer, quint32(std::max<qsizetype>(4, indexBytes))));
        require(d->vertices->create() && d->indices->create(), "Cannot create geometry buffers");
        for (qsizetype i = 0; i < offsets.size(); ++i) {
          const auto& g = d->scene->geometries[i];
          if (!g.vertices.isEmpty()) {
            batch->uploadStaticBuffer(d->vertices.get(), offsets[i], quint32(g.vertices.size() * sizeof(Vertex)), g.vertices.constData());
          }
        }

        if (!grid.isEmpty()) {
          batch->uploadStaticBuffer(d->vertices.get(), d->gridVertexOffset, quint32(grid.size() * sizeof(Vertex)), grid.constData());
        }

        if (!lightVertices.isEmpty()) {
          batch->uploadStaticBuffer(d->vertices.get(), lightVertexOffset, quint32(lightVertices.size() * sizeof(Vertex)), lightVertices.constData());
          batch->uploadStaticBuffer(d->indices.get(), lightIndexOffset * 4, quint32(lightIndices.size() * sizeof(quint32)), lightIndices.constData());
        }

        quint32 indexStart = 0;
        int modelDrawIndex = 0;
        QByteArray uniforms(d->stride * (d->scene->meshes.size() + d->scene->lights.size() + 1), 0);
        d->meshes.reserve(d->scene->meshes.size());
        d->order.reserve(d->scene->meshes.size());
        d->sortDistances.resize(d->scene->meshes.size());
        for (qsizetype i = 0; i < d->scene->meshes.size(); ++i) {
          const auto& m   = d->scene->meshes[i];
          const auto& mat = m.material;
          auto* bindings  = d->binding(rhi(), batch, mat);
          d->meshes.append({offsets.at(m.geometry), indexStart, quint32(m.indices.size()), (m.bounds.low + m.bounds.high) * .5f, bindings});
          if (!m.indices.isEmpty()) {
            batch->uploadStaticBuffer(d->indices.get(), indexStart * 4, quint32(m.indices.size() * 4), m.indices.constData());
          }

          indexStart += m.indices.size();
          DrawUniform u{};
          std::memcpy(u.model, m.transform.constData(), 64);
          u.tintGain[0] = mat.tint.x();
          u.tintGain[1] = mat.tint.y();
          u.tintGain[2] = mat.tint.z();
          u.tintGain[3] = mat.gain;
          if (!m_map && d->missingTextures.contains(mat.albedo)) {
            constexpr float palette[][3] = {{.76f, .79f, .82f}, {.68f, .73f, .77f}, {.78f, .74f, .68f}, {.66f, .73f, .71f}, {.74f, .70f, .75f}};
            std::copy_n(palette[modelDrawIndex % 5], 3, u.tintGain);
          }

          if (!m.indices.isEmpty()) {
            ++modelDrawIndex;
          }

          const uint32_t rgb = (uint32_t(i) * 0x9E3779 + 0x4F9CFF) & 0xFFFFFF;
          u.meshCutoff[0]    = (rgb >> 16) / 255.f;
          u.meshCutoff[1]    = ((rgb >> 8) & 255) / 255.f;
          u.meshCutoff[2]    = (rgb & 255) / 255.f;
          u.meshCutoff[3]    = mat.alphaCutoff;
          u.flags[0] =
              (mat.useLightmap ? 1 : 0) | (mat.useColors ? 2 : 0) | (d->scene->geometries[m.geometry].hasColors ? 4 : 0) | (mat.flipAlbedo ? 8 : 0) | (mat.flipLightmap ? 16 : 0);
          u.flags[2] = d->lastMip(mat.albedo);
          u.flags[3] = d->lastMip(mat.lightmap);
          std::memcpy(uniforms.data() + i * d->stride, &u, sizeof(u));
        }

        if (m_map && !d->scene->lights.isEmpty()) {
          const auto mat   = lightMaterial();
          d->lightBindings = d->binding(rhi(), batch, mat);
          d->lights.reserve(d->scene->lights.size());
          for (qsizetype i = 0; i < d->scene->lights.size(); ++i) {
            const auto& light = d->scene->lights[i];
            const bool point = light.type == LightType::Point;
            if (point) {
              d->lights.append({sphereVertexOffset, sphereIndexStart, sphereIndexCount});
            } else {
              d->lights.append({coneVertexOffset, coneIndexStart, coneIndexCount});
            }

            QMatrix4x4 transform;
            transform.translate(light.position);
            if (!point) {
              auto direction = light.direction;
              if (direction.lengthSquared() < 1e-12f) {
                direction = {0, 0, 1};
              }

              transform.rotate(QQuaternion::rotationTo({0, 0, 1}, direction.normalized()));
            }

            transform.scale(light.range);
            DrawUniform u{};
            std::memcpy(u.model, transform.constData(), 64);
            u.tintGain[0] = light.color.x();
            u.tintGain[1] = light.color.y();
            u.tintGain[2] = light.color.z();
            u.tintGain[3] = .35f;
            u.flags[1]    = 1;
            std::memcpy(uniforms.data() + (d->scene->meshes.size() + i) * d->stride, &u, sizeof(u));
          }
        }

        if (!m_map) {
          Material mat;
          mat.albedo      = "$white$";
          d->gridBindings = d->binding(rhi(), batch, mat);
          DrawUniform u{};
          QMatrix4x4 identity;
          std::memcpy(u.model, identity.constData(), 64);
          u.tintGain[0] = .22f;
          u.tintGain[1] = .27f;
          u.tintGain[2] = .32f;
          u.tintGain[3] = 1;
          u.flags[1]    = 1;
          std::memcpy(uniforms.data() + (d->scene->meshes.size() + d->scene->lights.size()) * d->stride, &u, sizeof(u));
        }

        batch->updateDynamicBuffer(d->draw.get(), 0, uniforms);
        d->unblended = m_map && mode != 0;
        for (const auto& mesh : d->scene->meshes) {
          d->pipeline(rhi(), m_target->pass.get(), mesh.material, wireframe, m_map && mode == 4, false, sampleCount());
        }

        if (showLights && !d->scene->lights.isEmpty()) {
          const auto mat = lightMaterial();
          d->pipeline(rhi(), m_target->pass.get(), mat, false, false, false, sampleCount(), true);
          d->pipeline(rhi(), m_target->pass.get(), mat, true, false, false, sampleCount(), true);
          d->pipeline(rhi(), m_target->pass.get(), mat, false, false, true, sampleCount(), true);
        }

        if (!m_map) {
          Material gridMaterial;
          d->pipeline(rhi(), m_target->pass.get(), gridMaterial, false, false, true, sampleCount());
        }

        cb->resourceUpdate(batch);
        batch   = nullptr;
        m_data  = std::move(d);
        m_scene = m_pending;
        m_pending.reset();
        m_paletteDirty = false;
        m_poseBounds   = m_scene->animation ? m_scene->animation->bounds() : m_scene->bounds;
        m_lastTick     = m_clock.nsecsElapsed();
        fit();
        if (sceneReady) {
          sceneReady(m_scene);
        }
      } catch (const std::exception& e) {
        if (batch) {
          batch->release();
        }

        m_pending.reset();
        if (failed) {
          failed(QString::fromUtf8(e.what()));
        }
      }
    }

    auto* batch = rhi()->nextResourceUpdateBatch();
    try {
      if (m_data) {
        auto& d            = *m_data;
        const float aspect = float(width()) / std::max(height(), 1);
        const auto b       = bounds(true);
        FrameUniform u{};
        const auto clip = rhi()->clipSpaceCorrMatrix() * camera.clip(b, aspect);
        std::memcpy(u.mvp, clip.constData(), 64);
        const auto light = camera.light();
        u.light[0]       = light.x();
        u.light[1]       = light.y();
        u.light[2]       = light.z();
        u.params[0]      = m_map ? mode : 5;
        batch->updateDynamicBuffer(d.frame.get(), 0, sizeof(u), &u);
        if (m_paletteDirty) {
          d.uploadPalette(batch);
          m_paletteDirty = false;
        }

        if (m_map) {
          visibleMeshes(*m_scene, camera.clip(b, aspect, true), d.order);
        } else {
          d.order.resize(d.meshes.size());
          std::iota(d.order.begin(), d.order.end(), 0);
        }

        visibleCount   = d.order.size();
        const auto eye = camera.eye();
        if (showLights && !d.lights.isEmpty()) {
          d.lightOrder.resize(d.lights.size());
          std::iota(d.lightOrder.begin(), d.lightOrder.end(), 0);
          d.lightDistances.resize(d.lights.size());
          for (qsizetype i = 0; i < d.lights.size(); ++i) {
            d.lightDistances[i] = (m_scene->lights[i].position - eye).lengthSquared();
          }

          std::sort(d.lightOrder.begin(), d.lightOrder.end(), [&](int a, int b) {
            return d.lightDistances[a] > d.lightDistances[b];
          });
        }

        if (m_map) {
          for (int i : d.order) {
            d.sortDistances[i] = (d.meshes[i].center - eye).lengthSquared();
          }

          std::sort(d.order.begin(), d.order.end(), [&](int a, int b) {
            const auto &ma = m_scene->meshes[a], &mb = m_scene->meshes[b];
            const auto &x = ma.material, &y = mb.material;
            const bool ta = x.blend != Blend::None, tb = y.blend != Blend::None;
            if (ta != tb) {
              return !ta;
            }

            if (x.sort != y.sort) {
              return x.sort < y.sort;
            }

            const float da = d.sortDistances[a], db = d.sortDistances[b];
            if (da != db) {
              return ta ? da > db : da < db;
            }

            if (!ta && x.name != y.name) {
              return x.name < y.name;
            }

            if (!ta && ma.sortGeometry != mb.sortGeometry) {
              return ma.sortGeometry < mb.sortGeometry;
            }

            return a < b;
          });
        }

        const bool solid = m_map && mode == 4;
        d.unblended      = m_map && mode != 0;
        for (int i : d.order) {
          d.pipeline(rhi(), m_target->pass.get(), m_scene->meshes[i].material, wireframe, solid, false, sampleCount());
        }

        if (showLights && !d.lights.isEmpty()) {
          const auto mat = lightMaterial();
          d.pipeline(rhi(), m_target->pass.get(), mat, false, false, false, sampleCount(), true);
          d.pipeline(rhi(), m_target->pass.get(), mat, true, false, false, sampleCount(), true);
          d.pipeline(rhi(), m_target->pass.get(), mat, false, false, true, sampleCount(), true);
        }

        if (!m_map && showGrid) {
          Material mat;
          d.pipeline(rhi(), m_target->pass.get(), mat, false, false, true, sampleCount());
        }
      }

      cb->beginPass(m_target->rt.get(), m_map ? QColor::fromRgbF(.055f, .07f, .09f, 1.f) : QColor::fromRgbF(.075f, .09f, .115f, 1.f), {m_map ? 0.f : 1.f, 0}, batch);
      batch = nullptr;
      if (m_data) {
        auto& d         = *m_data;
        const auto size = m_target->rt->pixelSize();
        cb->setViewport({0, 0, float(size.width()), float(size.height())});
        QRhiGraphicsPipeline* last = nullptr;
        for (int i : d.order) {
          const auto& g   = d.meshes[i];
          const auto& mat = m_scene->meshes[i].material;
          auto* p         = d.pipelines.at(d.pipelineKey(mat, wireframe, m_map && mode == 4, false)).get();
          if (p != last) {
            cb->setGraphicsPipeline(p);
            last = p;
          }

          const QRhiCommandBuffer::DynamicOffset offset(1, quint32(i * d.stride));
          cb->setShaderResources(g.bindings, 1, &offset);
          const QRhiCommandBuffer::VertexInput input(d.vertices.get(), g.vertexOffset);
          cb->setVertexInput(0, 1, &input, d.indices.get(), 0, QRhiCommandBuffer::IndexUInt32);
          cb->drawIndexed(g.indexCount, 1, g.indexStart);
        }

        if (showLights && !d.lightOrder.isEmpty()) {
          const auto mat = lightMaterial();
          for (int i : d.lightOrder) {
            const auto& g = d.lights[i];
            cb->setGraphicsPipeline(d.pipelines.at(d.pipelineKey(mat, false, false, false, true)).get());
            const QRhiCommandBuffer::DynamicOffset offset(1, quint32((d.meshes.size() + i) * d.stride));
            cb->setShaderResources(d.lightBindings, 1, &offset);
            const QRhiCommandBuffer::VertexInput input(d.vertices.get(), g.vertexOffset);
            cb->setVertexInput(0, 1, &input, d.indices.get(), 0, QRhiCommandBuffer::IndexUInt32);
            cb->drawIndexed(g.indexCount, 1, g.indexStart);
            cb->setGraphicsPipeline(d.pipelines.at(d.pipelineKey(mat, true, false, false, true)).get());
            cb->drawIndexed(g.indexCount, 1, g.indexStart);
          }

          cb->setGraphicsPipeline(d.pipelines.at(d.pipelineKey(mat, false, false, true, true)).get());
          for (int i : d.lightOrder) {
            const QRhiCommandBuffer::DynamicOffset offset(1, quint32((d.meshes.size() + i) * d.stride));
            cb->setShaderResources(d.lightBindings, 1, &offset);
            const QRhiCommandBuffer::VertexInput input(d.vertices.get(), d.crossVertexOffset);
            cb->setVertexInput(0, 1, &input);
            cb->draw(d.crossVertexCount);
          }
        }

        if (!m_map && showGrid) {
          Material mat;
          cb->setGraphicsPipeline(d.pipelines.at(d.pipelineKey(mat, false, false, true)).get());
          const QRhiCommandBuffer::DynamicOffset offset(1, quint32((d.meshes.size() + d.lights.size()) * d.stride));
          cb->setShaderResources(d.gridBindings, 1, &offset);
          const QRhiCommandBuffer::VertexInput input(d.vertices.get(), d.gridVertexOffset);
          cb->setVertexInput(0, 1, &input);
          cb->draw(d.gridVertexCount);
        }
      }

      cb->endPass();
    } catch (const std::exception& e) {
      if (batch) {
        batch->release();
      }

      if (failed) {
        failed(QString::fromUtf8(e.what()));
      }
    }
  }

  void RenderWidget::tick() {
    const auto now  = m_clock.nsecsElapsed();
    const double dt = double(now - m_lastTick) / 1e9;
    m_lastTick      = now;
    if (!isVisible() || !m_scene) {
      return;
    }

    bool dirty = camera.move(m_keys, float(dt));
    if (m_scene->animation) {
      try {
        if (m_scene->animation->update(dt)) {
          poseChanged();
          dirty = true;
        }
      } catch (const std::exception& e) {
        m_scene->animation->playing = false;
        if (failed) {
          failed(QString::fromUtf8(e.what()));
        }
      }
    }

    if (m_map || dirty) {
      update();
    }
  }

  void RenderWidget::releaseMouseCapture() {
    m_buttons = {};
    if (!m_mouseCaptured) {
      return;
    }

    m_mouseCaptured = false;
#ifdef Q_OS_WIN
    ClipCursor(nullptr);
    RAWINPUTDEVICE dev{0x01, 0x02, RIDEV_REMOVE, nullptr};
    RegisterRawInputDevices(&dev, 1, sizeof(dev));
#endif
    unsetCursor();
    if (QWidget::mouseGrabber() == this) {
      releaseMouse();
    }

    QCursor::setPos(m_restoreCursor);
  }

  void RenderWidget::mousePressEvent(QMouseEvent* e) {
    if (!m_scene || !(e->button() & (Qt::LeftButton | Qt::MiddleButton | Qt::RightButton))) {
      return;
    }

#ifdef Q_OS_WIN
    // Resolve the handle before grabMouse(), never while a grab is active.
    const auto hwnd = m_map ? reinterpret_cast<HWND>(winId()) : nullptr;
#endif
    setFocus(Qt::MouseFocusReason);
    const bool starting = !m_buttons;
    m_buttons |= e->button();
    if (starting) {
      m_restoreCursor = QCursor::pos();
      m_last          = e->position();
      if (m_map) {
        m_mouseCaptured = true;
        setCursor(Qt::BlankCursor);
        grabMouse();
        if (!m_mouseCaptured) {
          return;
        }

#ifdef Q_OS_WIN
        RAWINPUTDEVICE dev{0x01, 0x02, 0, hwnd};
        RegisterRawInputDevices(&dev, 1, sizeof(dev));
        RECT rect{};
        GetClientRect(hwnd, &rect);
        POINT origin{0, 0};
        ClientToScreen(hwnd, &origin);
        OffsetRect(&rect, origin.x, origin.y);
        ClipCursor(&rect);
#endif
      }
    }

    if (m_map && e->button() == Qt::RightButton) {
      camera.setFly(true);
    }
  }

  void RenderWidget::mouseReleaseEvent(QMouseEvent* e) {
    m_buttons &= ~e->button();
    if (!m_buttons) {
      releaseMouseCapture();
    }
  }

  void RenderWidget::applyMouseDelta(float dx, float dy, Qt::KeyboardModifiers mods) {
    if (!m_buttons) {
      return;
    }

    camera.drag(dx, dy, m_buttons, mods, height());
    update();
    if (changed) {
      changed();
    }
  }

  void RenderWidget::mouseMoveEvent(QMouseEvent* e) {
#ifdef Q_OS_WIN
    if (m_map && m_buttons) {
      return;
    }
#endif

    const auto delta = e->position() - m_last;
    m_last           = e->position();
    applyMouseDelta(delta.x(), delta.y(), e->modifiers());
  }

  void RenderWidget::wheelEvent(QWheelEvent* e) {
    if (m_scene) {
      camera.wheel(e->angleDelta().y() / 120.f);
      update();
      if (changed) {
        changed();
      }
    }

    e->accept();
  }

  void RenderWidget::keyPressEvent(QKeyEvent* e) {
    m_keys.insert(e->key());
    if (!e->isAutoRepeat() && keyAction) {
      keyAction(e->key(), e->modifiers());
    }

    e->accept();
  }

  void RenderWidget::keyReleaseEvent(QKeyEvent* e) {
    if (!e->isAutoRepeat()) {
      m_keys.remove(e->key());
    }

    e->accept();
  }

  void RenderWidget::focusOutEvent(QFocusEvent* e) {
    m_keys.clear();
    releaseMouseCapture();
    QRhiWidget::focusOutEvent(e);
  }

  bool RenderWidget::event(QEvent* e) {
    if (e->type() == QEvent::WindowDeactivate || e->type() == QEvent::Hide || e->type() == QEvent::UngrabMouse || e->type() == QEvent::WinIdChange) {
      m_keys.clear();
      releaseMouseCapture();
    }

    return QRhiWidget::event(e);
  }

  bool RenderWidget::nativeEvent(const QByteArray& type, void* message, qintptr* result) {
#ifdef Q_OS_WIN
    const auto* msg = static_cast<MSG*>(message);
    if (m_mouseCaptured && (msg->message == WM_CAPTURECHANGED || msg->message == WM_CANCELMODE)) {
      releaseMouseCapture();
    }

    if (msg->message == WM_INPUT && m_map && m_mouseCaptured && m_buttons) {
      RAWINPUT input{};
      UINT size = sizeof(input);
      if (GetRawInputData(reinterpret_cast<HRAWINPUT>(msg->lParam), RID_INPUT, &input, &size, sizeof(RAWINPUTHEADER)) != UINT(-1) && input.header.dwType == RIM_TYPEMOUSE &&
          !(input.data.mouse.usFlags & MOUSE_MOVE_ABSOLUTE)) {
        applyMouseDelta(input.data.mouse.lLastX, input.data.mouse.lLastY, QApplication::keyboardModifiers());
      }
    }
#endif

    return QRhiWidget::nativeEvent(type, message, result);
  }

  void RenderWidget::dragEnterEvent(QDragEnterEvent* e) {
    if (e->mimeData()->hasUrls()) {
      e->acceptProposedAction();
    }
  }

  void RenderWidget::dropEvent(QDropEvent* e) {
    const auto urls = e->mimeData()->urls();
    if (urls.isEmpty()) {
      return;
    }

    QString path = urls.first().toLocalFile();
    for (const auto& u : urls) {
      if (u.toLocalFile().endsWith(m_map ? ".ds2" : ".ds2md", Qt::CaseInsensitive)) {
        path = u.toLocalFile();
        break;
      }
    }

    releaseMouseCapture();
    if (fileDropped) {
      fileDropped(path);
    }

    e->acceptProposedAction();
  }
} // namespace ds2
