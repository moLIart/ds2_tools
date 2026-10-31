#include "viewer_window.h"
#include "glb_export.h"
#include <QAction>
#include <QApplication>
#include <QComboBox>
#include <QDebug>
#include <QDir>
#include <QDirIterator>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFileInfo>
#include <QLabel>
#include <QMessageBox>
#include <QMouseEvent>
#include <QSignalBlocker>
#include <QSlider>
#include <QStatusBar>
#include <QStyle>
#include <QStyleOptionSlider>
#include <QTimer>
#include <QToolBar>
#include <algorithm>

namespace ds2 {
namespace {
class TimelineSlider final : public QSlider {
  public:
    explicit TimelineSlider(QWidget *parent) : QSlider(Qt::Horizontal, parent) {}

  protected:
    void mousePressEvent(QMouseEvent *event) override {
        if (event->button() != Qt::LeftButton) {
            QSlider::mousePressEvent(event);
            return;
        }

        setSliderDown(true);
        seek(event->position().x());
        event->accept();
    }
    void mouseMoveEvent(QMouseEvent *event) override {
        if (isSliderDown()) {
            seek(event->position().x());
            event->accept();
        } else
            QSlider::mouseMoveEvent(event);
    }
    void mouseReleaseEvent(QMouseEvent *event) override {
        if (event->button() == Qt::LeftButton && isSliderDown()) {
            seek(event->position().x());
            setSliderDown(false);
            event->accept();
        } else
            QSlider::mouseReleaseEvent(event);
    }

  private:
    void seek(qreal x) {
        QStyleOptionSlider option;
        initStyleOption(&option);
        const int handle = style()->pixelMetric(QStyle::PM_SliderLength, &option, this);
        setValue(QStyle::sliderValueFromPosition(minimum(), maximum(), qRound(x) - handle / 2,
                                                 std::max(1, width() - handle), option.upsideDown));
    }
};
} // namespace
QString findGameRoot(const QString &path, const QString &explicitRoot) {
    if (!explicitRoot.isEmpty())
        return QDir(explicitRoot).absolutePath();

    for (const auto &start :
         {QFileInfo(path).absolutePath(), QDir::currentPath(), QApplication::applicationDirPath()}) {
        QDir d(start);
        do {
            if (QDir(d.filePath("gameres")).exists())
                return d.absolutePath();
        } while (d.cdUp());
    }

    return QDir::currentPath();
}
QAction *ViewerWindow::action(const QString &text, const QKeySequence &shortcut,
                              std::function<void()> callback, bool check) {
    auto *a = m_toolbar->addAction(text);
    a->setCheckable(check);
    if (!shortcut.isEmpty())
        a->setShortcut(shortcut);

    connect(a, &QAction::triggered, this, [callback = std::move(callback)] { callback(); });
    return a;
}
std::shared_ptr<Animation> ViewerWindow::animation() const {
    return m_view->scene() ? m_view->scene()->animation : nullptr;
}
ViewerWindow::ViewerWindow(bool map, const QString &root, const QString &path, int mode,
                           const QString &screenshot, bool debug)
    : m_map(map), m_root(root), m_screenshot(screenshot) {
    resize(map ? 1280 : 1200, map ? 850 : 800);
    setMinimumSize(map ? 760 : 640, map ? 480 : 420);
    setWindowTitle(map ? "DS2 Map Viewer" : "DS2MD Model Viewer");
    m_view = new RenderWidget(map, this);
    m_view->mode = mode;
    m_view->setDebugLayerEnabled(debug);
    m_view->setObjectName("renderFrame");
    setCentralWidget(m_view);
    m_toolbar = addToolBar("Viewer");
    m_toolbar->setObjectName("viewerToolbar");
    m_toolbar->setMovable(false);
    action("Open...", QKeySequence::Open, [this] { open(); });
    if (!map) {
        m_export = action("Export GLB...", QKeySequence("Ctrl+Shift+E"), [this] { exportModelGlb(); });
        m_export->setObjectName("exportGlb");
        m_export->setToolTip("Export the model, all animation clips, and embedded textures");
        m_export->setEnabled(false);
    }

    if (map) {
        m_toolbar->addWidget(new QLabel(QStringLiteral("Map: "), m_toolbar));
        m_maps = new QComboBox(m_toolbar);
        m_maps->setObjectName(QStringLiteral("mapSelection"));
        m_maps->setAccessibleName(QStringLiteral("Map"));
        m_maps->setPlaceholderText(QStringLiteral("Select map"));
        m_maps->setMinimumContentsLength(20);
        m_maps->setMaximumWidth(300);
        m_toolbar->addWidget(m_maps);
        connect(m_maps, &QComboBox::activated, this,
                [this](int index) { load(m_maps->itemData(index).toString()); });
    }

    action("Fit", QKeySequence(map ? "Home" : "F"), [this] { m_view->fit(); });
    m_wire = action(
        "Wire", QKeySequence(map ? "G" : "W"),
        [this] {
            m_view->wireframe = m_wire->isChecked();
            m_view->update();
        },
        true);
    if (map) {
        m_fly = action(
            "Fly", QKeySequence("F"),
            [this] {
                m_view->camera.setFly(m_fly->isChecked());
                m_view->update();
                sync();
            },
            true);
        m_modes = new QComboBox(m_toolbar);
        m_modes->setObjectName("renderMode");
        m_modes->addItems({"Combined", "Albedo", "Lightmap", "Vertex colors", "Mesh colors"});
        m_modes->setCurrentIndex(mode);
        m_toolbar->addWidget(m_modes);
        connect(m_modes, &QComboBox::currentIndexChanged, this, [this](int i) {
            m_view->mode = i;
            m_view->update();
        });
        for (int i = 0; i < 5; ++i) {
            auto *a = new QAction(this);
            a->setShortcut(QKeySequence(QString::number(i + 1)));
            addAction(a);
            connect(a, &QAction::triggered, this, [this, i] { m_modes->setCurrentIndex(i); });
        }

        m_toolbar->addSeparator();
        m_toolbar->addWidget(new QLabel("Distance: ", m_toolbar));
        m_distance = new QDoubleSpinBox(m_toolbar);
        m_distance->setObjectName("drawDistance");
        m_distance->setRange(0, 1e8);
        m_distance->setDecimals(1);
        m_distance->setSpecialValueText("Auto");
        m_distance->setSingleStep(100);
        m_distance->setMaximumWidth(120);
        m_toolbar->addWidget(m_distance);
        connect(m_distance, &QDoubleSpinBox::valueChanged, this, [this](double v) {
            if (v > 0 && v < 1) {
                m_distance->setValue(1);
                return;
            }

            m_view->camera.drawDistance = float(v);
            m_view->update();
        });
        auto distance = [this](float factor) {
            auto &c = m_view->camera;
            auto scene = m_view->scene();
            if (!scene)
                return;

            const float d = c.drawDistance > 0 ? c.drawDistance : c.planes(scene->bounds, false).second;
            m_distance->setValue(std::clamp(d * factor, 1.f, 1e8f));
        };
        auto *up = new QAction(this), *down = new QAction(this), *automatic = new QAction(this);
        up->setShortcut(QKeySequence("PgUp"));
        down->setShortcut(QKeySequence("PgDown"));
        automatic->setShortcut(QKeySequence("End"));
        addActions({up, down, automatic});
        connect(up, &QAction::triggered, this, [distance] { distance(1.25f); });
        connect(down, &QAction::triggered, this, [distance] { distance(.8f); });
        connect(automatic, &QAction::triggered, this, [this] { m_distance->setValue(0); });
        const QDir mapsDir(QDir(root).filePath(QStringLiteral("gameres/maps")));
        QStringList maps;
        QDirIterator it(mapsDir.path(), {"*.ds2"}, QDir::Files, QDirIterator::Subdirectories);
        while (it.hasNext())
            maps.append(QFileInfo(it.next()).absoluteFilePath());

        maps.sort();
        for (const auto &mapPath : maps)
            m_maps->addItem(mapsDir.relativeFilePath(mapPath), mapPath);

        m_maps->setCurrentIndex(-1);
    } else {
        auto *grid = action(
            "Grid", QKeySequence("G"),
            [this] {
                m_view->showGrid = !m_view->showGrid;
                m_view->update();
            },
            true);
        grid->setChecked(true);
        action("Light", QKeySequence("L"), [this] {
            m_view->camera.resetLight();
            m_view->update();
        });
        m_toolbar->addSeparator();
        m_clips = new QComboBox(m_toolbar);
        m_clips->setObjectName("animationClip");
        m_clips->setMinimumContentsLength(17);
        m_clips->setMaximumWidth(260);
        m_toolbar->addWidget(m_clips);
        connect(m_clips, &QComboBox::currentIndexChanged, this, [this](int i) {
            if (!m_seeking)
                selectClip(i - 1);
        });
        action("<", QKeySequence("["), [this] { cycleClip(-1); });
        action(">", QKeySequence("]"), [this] { cycleClip(1); });
        m_play = action("Play", QKeySequence("Space"), [this] { play(); });
        m_bind = action("Bind", QKeySequence("B"), [this] { selectClip(-1); });
        m_repeat = action(
            "Repeat", QKeySequence("R"),
            [this] {
                if (auto a = animation())
                    a->loop = m_repeat->isChecked();
            },
            true);
        m_repeat->setChecked(true);
        m_speed = new QDoubleSpinBox(m_toolbar);
        m_speed->setObjectName("animationSpeed");
        m_speed->setRange(.125, 8);
        m_speed->setDecimals(3);
        m_speed->setValue(1);
        m_speed->setSuffix("x");
        m_speed->setSingleStep(.125);
        m_speed->setMaximumWidth(90);
        m_toolbar->addWidget(m_speed);
        connect(m_speed, &QDoubleSpinBox::valueChanged, this, [this](double v) {
            if (auto a = animation())
                a->speed = v;
        });
        auto *slower = new QAction(this), *faster = new QAction(this);
        slower->setShortcuts({QKeySequence("-"), QKeySequence("Num+-")});
        faster->setShortcuts({QKeySequence("="), QKeySequence("+"), QKeySequence("Num++")});
        addActions({slower, faster});
        connect(slower, &QAction::triggered, this, [this] { m_speed->setValue(m_speed->value() * .5); });
        connect(faster, &QAction::triggered, this, [this] { m_speed->setValue(m_speed->value() * 2); });
        addToolBar(Qt::BottomToolBarArea, new QToolBar("Playback", this));
    }

    action("Controls", {}, [this] {
        m_view->releaseMouseCapture();
        QMessageBox::information(
            this, "Controls",
            m_map ? "LMB orbit | MMB pan | Wheel zoom/speed | RMB fly-look\nF fly/orbit | WASD, Q/E move | "
                    "Shift fast\n1 combined | 2 albedo | 3 lightmap | 4 vertex colors | 5 mesh colors\nG "
                    "wireframe | Home fit | Map list select | Ctrl+O open\nPgUp/PgDown draw distance | End "
                    "automatic | Esc close"
                  : "LMB orbit | Shift+LMB light | RMB/MMB pan | Wheel zoom\nF fit | L reset light | W "
                    "wireframe | G grid | Ctrl+O open | Ctrl+Shift+E export GLB\nSpace play/pause | [ / ] clips | B bind/stored "
                    "geometry\nArrows seek 0.1s | Shift+Left/Right or Up/Down seek 1s | Home start\n- / + "
                    "speed | R repeat | Esc close");
    });
    auto *closeAction = new QAction(this);
    closeAction->setShortcut(QKeySequence("Esc"));
    addAction(closeAction);
    connect(closeAction, &QAction::triggered, this, &QWidget::close);
    m_stats = new QLabel(this);
    statusBar()->addPermanentWidget(m_stats);
    statusBar()->showMessage(map ? "Open or drop a .ds2 map" : "Open or drop a .ds2md model");
    if (!map) {
        auto *bar = findChildren<QToolBar *>().last();
        bar->setObjectName("playbackToolbar");
        bar->setMovable(false);
        m_timeline = new TimelineSlider(bar);
        m_timeline->setObjectName("animationTimeline");
        m_timeline->setRange(0, 10000);
        m_timeline->setMinimumWidth(200);
        m_timeline->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
        bar->addWidget(m_timeline);
        m_time = new QLabel("0.00 / 0.00 s", bar);
        m_time->setMinimumWidth(140);
        bar->addWidget(m_time);
        connect(m_timeline, &QSlider::sliderPressed, this, [this] {
            if (auto a = animation()) {
                m_seeking = true;
                m_resume = a->playing;
                a->playing = false;
            }
        });
        connect(m_timeline, &QSlider::valueChanged, this, [this](int v) {
            if (m_seeking)
                if (auto a = animation())
                    seek(a->duration() * v / 10000.);
        });
        connect(m_timeline, &QSlider::sliderReleased, this, [this] {
            if (auto a = animation())
                a->playing = m_resume;

            m_seeking = false;
            sync();
        });
    }

    m_view->keyAction = [this](int key, Qt::KeyboardModifiers mods) {
        if (!m_map) {
            if (key == Qt::Key_Home)
                seek(0);
            else if (key == Qt::Key_Left || key == Qt::Key_Right || key == Qt::Key_Up ||
                     key == Qt::Key_Down) {
                if (auto a = animation())
                    seek(a->time() +
                         ((key == Qt::Key_Left || key == Qt::Key_Down) ? -1 : 1) *
                             ((key == Qt::Key_Up || key == Qt::Key_Down || mods & Qt::ShiftModifier) ? 1.
                                                                                                     : .1));
            }
        }
    };
    m_view->sceneReady = [this](const std::shared_ptr<Scene> &scene) {
        m_loaded = true;
        setWindowTitle(QFileInfo(scene->path).fileName() +
                       (m_map ? " - DS2 Map Viewer" : " - DS2MD Model Viewer"));
        qsizetype vertices = 0, triangles = 0;
        for (const auto &g : scene->geometries)
            vertices += g.vertices.size();

        for (const auto &m : scene->meshes)
            triangles += m.indices.size() / 3;

        statusBar()->setToolTip(scene->path + '\n' + scene->note + '\n' + scene->warnings.join('\n'));
        statusBar()->showMessage(QString("%1 | %2 vertices | %3 triangles%4")
                                     .arg(scene->name)
                                     .arg(vertices)
                                     .arg(triangles)
                                     .arg(scene->warnings.isEmpty() ? QString()
                                                                    : QString(" | %1 warnings: %2")
                                                                          .arg(scene->warnings.size())
                                                                          .arg(scene->warnings.first())));
        for (const auto &w : scene->warnings)
            qWarning().noquote() << QFileInfo(scene->path).fileName() + ": " + w;

        if (m_maps) {
            int index = m_maps->findData(scene->path);
            if (index < 0) {
                const QDir mapsDir(QDir(m_root).filePath(QStringLiteral("gameres/maps")));
                m_maps->addItem(mapsDir.relativeFilePath(scene->path), scene->path);
                index = m_maps->count() - 1;
            }

            m_maps->setCurrentIndex(index);
        }

        if (m_clips) {
            QSignalBlocker blocker(m_clips);
            m_clips->clear();
            m_clips->addItem(scene->sampledAnimation ? "Stored geometry" : "Bind pose");
            if (auto a = animation())
                for (int i = 0; i < a->clipCount(); ++i)
                    m_clips->addItem(a->clipName(i));
        }

        if (m_bind)
            m_bind->setText(scene->sampledAnimation ? "Stored" : "Bind");

        sync();
        m_view->setFocus();
        if (!m_screenshot.isEmpty())
            QTimer::singleShot(200, this, [this] {
                if (!m_loaded)
                    return;

                auto image = m_view->grabFramebuffer();
                if (image.isNull() || !image.save(m_screenshot)) {
                    error("Cannot save screenshot: " + m_screenshot);
                    return;
                }

                qInfo().noquote() << "Saved" << m_screenshot;
                QApplication::exit(0);
            });
    };
    m_view->failed = [this](const QString &s) { error(s); };
    m_view->fileDropped = [this](const QString &p) { load(p); };
    m_view->changed = [this] { sync(); };
    auto *uiTimer = new QTimer(this);
    uiTimer->setInterval(100);
    connect(uiTimer, &QTimer::timeout, this, [this] { sync(); });
    uiTimer->start();
    if (!path.isEmpty())
        load(path);
    else if (m_maps && m_maps->count() > 0)
        load(m_maps->itemData(0).toString());
}
void ViewerWindow::error(const QString &message) {
    qCritical().noquote() << message;
    statusBar()->showMessage("Error: " + message);
    if (m_maps) {
        const auto &scene = m_view->scene();
        m_maps->setCurrentIndex(scene ? m_maps->findData(scene->path) : -1);
    }

    if (!m_screenshot.isEmpty())
        QTimer::singleShot(0, [] { QApplication::exit(1); });
}
void ViewerWindow::load(const QString &path) {
    m_view->releaseMouseCapture();
    m_pending = QFileInfo(path).absoluteFilePath();
    statusBar()->showMessage("Loading " + m_pending + " ...");
    QTimer::singleShot(20, this, [this, path = m_pending] {
        if (path != m_pending)
            return;

        try {
            auto scene = std::make_shared<Scene>(m_map ? loadMap(path, m_root) : loadModel(path, m_root));
            m_view->setScene(std::move(scene));
        } catch (const std::exception &e) {
            error(QString("Cannot open %1: %2").arg(path, QString::fromUtf8(e.what())));
        }
    });
}
void ViewerWindow::open() {
    m_view->releaseMouseCapture();
    const auto current = m_view->scene();
    const auto path = QFileDialog::getOpenFileName(this, m_map ? "Open DS2 map" : "Open DS2MD model",
                                                   current ? QFileInfo(current->path).absolutePath() : m_root,
                                                   m_map ? "DS2 maps (*.ds2)" : "DS2MD models (*.ds2md)");
    if (!path.isEmpty())
        load(path);

    m_view->setFocus();
}
void ViewerWindow::exportModelGlb() {
    const auto scene = m_view->scene();
    if (!scene || m_map)
        return;

    m_view->releaseMouseCapture();
    const QFileInfo source(scene->path);
    QString path = QFileDialog::getSaveFileName(
        this, "Export model and all animations",
        source.absoluteDir().filePath(source.completeBaseName() + ".glb"), "Binary glTF (*.glb)");
    if (path.isEmpty())
        return;

    if (QFileInfo(path).suffix().isEmpty())
        path += ".glb";

    QStringList warnings;
    QApplication::setOverrideCursor(Qt::WaitCursor);
    try {
        warnings = exportGlb(*scene, path);
    } catch (const std::exception &e) {
        QApplication::restoreOverrideCursor();
        const QString message = QString("Cannot export %1: %2").arg(path, QString::fromUtf8(e.what()));
        error(message);
        QMessageBox::critical(this, "GLB export failed", message);
        m_view->setFocus();
        return;
    }

    QApplication::restoreOverrideCursor();
    statusBar()->showMessage(
        QString("Exported %1%2").arg(path, warnings.isEmpty() ? QString()
                                   : QString(" | %1 warnings").arg(warnings.size())), 10000);
    qInfo().noquote() << "Exported" << path;
    if (!warnings.isEmpty()) {
        for (const auto &warning : warnings)
            qWarning().noquote() << warning;

        QMessageBox message(QMessageBox::Warning, "GLB exported with warnings",
                            "Saved " + path + "\n\n" + warnings.first(), QMessageBox::Ok, this);
        message.setDetailedText(warnings.join('\n'));
        message.exec();
    }

    m_view->setFocus();
}
void ViewerWindow::selectClip(int index) {
    if (auto a = animation()) {
        try {
            a->select(index);
            m_view->poseChanged();
            m_view->fit();
            sync();
        } catch (const std::exception &e) {
            error(QString::fromUtf8(e.what()));
        }
    }
}
void ViewerWindow::cycleClip(int direction) {
    if (auto a = animation()) {
        const int selection = (a->clipIndex() + 1 + direction + a->clipCount() + 1) % (a->clipCount() + 1);
        selectClip(selection - 1);
    }
}
void ViewerWindow::play() {
    if (auto a = animation()) {
        if (a->clipIndex() < 0) {
            if (a->clipCount())
                selectClip(0);

            return;
        }

        if (!a->playing && a->time() >= a->duration())
            seek(0);

        a->playing = !a->playing;
        sync();
    }
}
void ViewerWindow::seek(double time) {
    if (auto a = animation()) {
        try {
            a->seek(time);
            m_view->poseChanged();
            sync();
        } catch (const std::exception &e) {
            error(QString::fromUtf8(e.what()));
        }
    }
}
void ViewerWindow::sync() {
    const auto scene = m_view->scene();
    if (m_fly) {
        QSignalBlocker b(m_fly);
        m_fly->setChecked(m_view->camera.fly);
    }

    m_stats->setText(scene ? QString("%1 | %2/%3 meshes | %4 FPS%5")
                                 .arg(scene->version)
                                 .arg(m_view->visibleCount)
                                 .arg(scene->meshes.size())
                                 .arg(m_view->fps(), 0, 'f', 1)
                                 .arg(m_map ? QString(" | %1 | speed %2")
                                                  .arg(m_view->camera.fly ? "Fly" : "Orbit")
                                                  .arg(m_view->camera.speed, 0, 'f', 1)
                                            : QString())
                           : QString());
    if (!m_map) {
        auto a = animation();
        m_export->setEnabled(bool(scene));
        const bool animated = bool(a);
        m_clips->setEnabled(animated);
        m_play->setEnabled(animated);
        m_repeat->setEnabled(animated);
        m_speed->setEnabled(animated);
        m_timeline->setEnabled(animated && a->clipIndex() >= 0);
        if (!a)
            return;

        {
            QSignalBlocker b(m_clips);
            m_clips->setCurrentIndex(a->clipIndex() + 1);
        }
        {
            QSignalBlocker b(m_speed);
            m_speed->setValue(a->speed);
        }
        {
            QSignalBlocker b(m_repeat);
            m_repeat->setChecked(a->loop);
        }
        m_play->setText(a->playing ? "Pause" : "Play");
        m_time->setText(QString("%1 / %2 s").arg(a->time(), 0, 'f', 2).arg(a->duration(), 0, 'f', 2));
        if (!m_seeking) {
            QSignalBlocker b(m_timeline);
            m_timeline->setValue(a->duration() > 0 ? int(a->time() / a->duration() * 10000) : 0);
        }
    }
}
} // namespace ds2
