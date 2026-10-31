#pragma once
#include "render_widget.h"
#include <QMainWindow>
class QAction;
class QComboBox;
class QDoubleSpinBox;
class QLabel;
class QSlider;
class QToolBar;

namespace ds2 {
class ViewerWindow final : public QMainWindow {
  public:
    ViewerWindow(bool map, const QString &root, const QString &path, int mode, const QString &screenshot,
                 bool debug);

  private:
    bool m_map, m_loaded = false, m_seeking = false, m_resume = false;
    QString m_root, m_screenshot, m_pending;
    QStringList m_maps;
    RenderWidget *m_view;
    QToolBar *m_toolbar;
    QLabel *m_stats, *m_time = nullptr;
    QComboBox *m_modes = nullptr, *m_clips = nullptr;
    QDoubleSpinBox *m_distance = nullptr, *m_speed = nullptr;
    QSlider *m_timeline = nullptr;
    QAction *m_fly = nullptr, *m_wire, *m_play = nullptr, *m_repeat = nullptr, *m_bind = nullptr;
    QAction *m_export = nullptr;
    QAction *action(const QString &text, const QKeySequence &shortcut, std::function<void()> callback,
                    bool check = false);
    std::shared_ptr<Animation> animation() const;
    void load(const QString &path);
    void open();
    void exportModelGlb();
    void changeMap(int delta);
    void selectClip(int index);
    void cycleClip(int direction);
    void play();
    void seek(double time);
    void sync();
    void error(const QString &message);
};
QString findGameRoot(const QString &path, const QString &explicitRoot);
} // namespace ds2
