#pragma once
#include "camera.h"
#include <functional>
#include <QElapsedTimer>
#include <QRhiWidget>

namespace ds2 {
  class RenderWidget final : public QRhiWidget {
  public:
    explicit RenderWidget(bool map, QWidget* parent = nullptr);
    ~RenderWidget() override;
    Camera camera;
    bool wireframe = false, showGrid = true, showLights = true, showModels = true;
    int mode = 0, visibleCount = 0;
    std::function<void(std::shared_ptr<Scene>)> sceneReady;
    std::function<void(QString)> failed;
    std::function<void(int, Qt::KeyboardModifiers)> keyAction;
    std::function<void(QString)> fileDropped;
    std::function<void()> changed;
    std::shared_ptr<Scene> scene() const;
    void setScene(std::shared_ptr<Scene> scene);
    void setSelectedMeshes(QVector<int> meshes);
    void fit();
    void releaseMouseCapture();
    void poseChanged();

    double fps() const {
      return m_fps;
    }

  protected:
    void initialize(QRhiCommandBuffer*) override;
    void render(QRhiCommandBuffer*) override;
    void releaseResources() override;
    void mousePressEvent(QMouseEvent*) override;
    void mouseReleaseEvent(QMouseEvent*) override;
    void mouseMoveEvent(QMouseEvent*) override;
    void wheelEvent(QWheelEvent*) override;
    void keyPressEvent(QKeyEvent*) override;
    void keyReleaseEvent(QKeyEvent*) override;
    void focusOutEvent(QFocusEvent*) override;
    bool event(QEvent*) override;
    bool nativeEvent(const QByteArray&, void*, qintptr*) override;
    void dragEnterEvent(QDragEnterEvent*) override;
    void dropEvent(QDropEvent*) override;

  private:
    struct Data;
    std::unique_ptr<Data> m_data;
    std::shared_ptr<Scene> m_pending, m_scene;
    QVector<int> m_selectedMeshes;
    QVector<Vertex> m_selectionGizmos;
    bool m_map, m_paletteDirty = true, m_gizmosDirty = true;
    Bounds m_poseBounds;
    QSet<int> m_keys;
    Qt::MouseButtons m_buttons;
    bool m_mouseCaptured = false;
    QPointF m_last;
    QPoint m_restoreCursor;
    QElapsedTimer m_clock, m_fpsClock;
    qint64 m_lastTick = 0;
    int m_frames      = 0;
    double m_fps      = 0;
    struct Target;
    std::unique_ptr<Target> m_target;
    QRhi* m_owner = nullptr;
    void tick();
    Bounds bounds(bool grid = false) const;
    void applyMouseDelta(float dx, float dy, Qt::KeyboardModifiers mods);
  };
} // namespace ds2
