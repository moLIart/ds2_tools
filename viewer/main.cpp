#include "viewer_window.h"
#include "glb_export.h"
#include <QApplication>
#include <QCommandLineParser>
#include <QSurfaceFormat>
#include <cstdio>
#include <exception>

int main(int argc, char **argv) {
    Q_INIT_RESOURCE(viewer_shaders);
#ifdef DS2_MAP_VIEWER
    constexpr bool map = true;
#else
    constexpr bool map = false;
#endif
    QSurfaceFormat format = QSurfaceFormat::defaultFormat();
    format.setSwapInterval(1);
    QSurfaceFormat::setDefaultFormat(format);
    QApplication app(argc, argv);
    QApplication::setApplicationName(map ? "ds2_view" : "ds2md_view");
    QApplication::setApplicationVersion("1.0");
    QCommandLineParser parser;
    parser.setApplicationDescription(map ? "Qt6 QRhi DS2 map viewer"
                                         : "Qt6 QRhi DS2MD model and animation viewer");
    parser.addHelpOption();
    parser.addVersionOption();
    parser.addPositionalArgument("file", map ? "Optional .ds2 map" : "Optional .ds2md model");
    parser.addOption({"game-root", "Game installation containing gameres", "directory"});
    parser.addOption({"screenshot", "Save the rendering frame as an image and exit", "file"});
    parser.addOption({"debug-rhi", "Enable the graphics API validation/debug layer"});
    if (map)
        parser.addOption({"mode", "combined, albedo, lightmap, colors, or meshes", "mode", "combined"});
    else
        parser.addOption({"export-glb", "Export the model and all animation clips to GLB and exit", "file"});
    parser.process(app);
    const auto args = parser.positionalArguments();
    if (args.size() > 1)
        parser.showHelp(1);
    const auto path = args.isEmpty() ? QString() : args.first();
    const QStringList modes = {"combined", "albedo", "lightmap", "colors", "meshes"};
    const int mode = map ? modes.indexOf(parser.value("mode")) : 0;
    if (mode < 0) {
        std::fprintf(stderr, "Invalid render mode\n");
        return 1;
    }
    if (!map && path.isEmpty() && parser.isSet("screenshot")) {
        std::fprintf(stderr, "A model path is required for --screenshot\n");
        return 1;
    }
    if (!map && parser.isSet("export-glb")) {
        if (path.isEmpty() || parser.isSet("screenshot")) {
            std::fprintf(stderr, "--export-glb requires a model path and cannot be combined with --screenshot\n");
            return 1;
        }
        try {
            const auto scene = ds2::loadModel(path, ds2::findGameRoot(path, parser.value("game-root")));
            const QString destination = parser.value("export-glb");
            const auto warnings = ds2::exportGlb(scene, destination);
            for (const auto &warning : warnings)
                std::fprintf(stderr, "Warning: %s\n", warning.toUtf8().constData());
            std::printf("Saved %s\n", destination.toUtf8().constData());
            return 0;
        } catch (const std::exception &e) {
            std::fprintf(stderr, "GLB export failed: %s\n", e.what());
            return 1;
        }
    }
    ds2::ViewerWindow window(map, ds2::findGameRoot(path, parser.value("game-root")), path, mode,
                             parser.value("screenshot"), parser.isSet("debug-rhi"));
    window.show();
    return app.exec();
}
