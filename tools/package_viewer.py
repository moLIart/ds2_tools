"""Package native Qt viewer sources; never include game assets or build output.

Run: python tools/package_viewer.py [--output viewer.zip]
The recipient builds with Qt 6.11 and a compatible C++ toolchain.
Python is used only by this packaging utility, not by either viewer.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZIP_DEFLATED, ZipFile


SOURCE_DIR = Path(__file__).resolve().parent.parent
RUNTIME_FILES = (
    'CMakeLists.txt',
    'build.ps1',
    'install.ps1',
    'viewer/main.cpp',
    'viewer/scene.h',
    'viewer/camera.h',
    'viewer/map_loader.cpp',
    'viewer/model_loader.cpp',
    'viewer/model_animation.cpp',
    'viewer/model_animation.h',
    'viewer/material.cpp',
    'viewer/material.h',
    'viewer/glb_export.h',
    'viewer/glb_export.cpp',
    'viewer/glb_animation.cpp',
    'viewer/texture.cpp',
    'viewer/texture.h',
    'viewer/render_widget.cpp',
    'viewer/render_widget.h',
    'viewer/viewer_window.cpp',
    'viewer/viewer_window.h',
    'viewer/shaders/scene.vert',
    'viewer/shaders/scene.frag',
)
ARCHIVE_DIRECTORY = 'ds2_viewers/ds2_research'
DEFAULT_OUTPUT = SOURCE_DIR / 'dist' / 'ds2-viewers-source.zip'


def build_archive(output: Path) -> Path:
    output = output.expanduser().resolve()
    if output.suffix.lower() != '.zip':
        raise ValueError('The output filename must end in .zip.')
    for filename in RUNTIME_FILES:
        if not (SOURCE_DIR / filename).is_file():
            raise FileNotFoundError(f'Required viewer source is missing: {SOURCE_DIR / filename}')
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=output.parent, prefix='viewer-', suffix='.tmp', delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with ZipFile(temporary_path, 'w', compression=ZIP_DEFLATED, compresslevel=9) as archive:
            for filename in RUNTIME_FILES:
                archive.write(SOURCE_DIR / filename, f'{ARCHIVE_DIRECTORY}/{filename}')
        temporary_path.replace(output)
    finally:
        temporary_path.unlink(missing_ok=True)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description='Pack Qt6 C++ viewer sources only; no game resources or binaries.')
    parser.add_argument('-o', '--output', type=Path, default=DEFAULT_OUTPUT,
                        help='ZIP filename (default: ds2_research/dist/ds2-viewers-source.zip)')
    args = parser.parse_args()
    try:
        output = build_archive(args.output)
    except (OSError, ValueError) as error:
        parser.exit(1, f'Cannot pack viewers: {error}\n')
    print(f'Created: {output}')
    print(f'Archive size: {output.stat().st_size} bytes')
    print('Included:')
    for filename in RUNTIME_FILES:
        print(f'  {ARCHIVE_DIRECTORY}/{filename}')
    print('No models, textures, game DLLs, Qt binaries or build output included.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
