"""Create a source-only ZIP of the viewer; never include game resources.

Run from any directory:
    python path/to/ds2_research/package_viewer.py
    python path/to/ds2_research/package_viewer.py --output viewer.zip

The recipient installs Python and the packages in requirements-viewer.txt.
Models, textures, game DLLs, research documents and bytecode are not included.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZIP_DEFLATED, ZipFile


SOURCE_DIR = Path(__file__).resolve().parent
RUNTIME_FILES = (
    'view_ds2md.py',
    'ds2md_scene.py',
    'ds2md_animation.py',
    'inspect_ds2md.py',
    'ds2md_legacy.py',
    'requirements-viewer.txt',
)
ARCHIVE_DIRECTORY = 'ds2md_viewer/ds2_research'
DEFAULT_OUTPUT = SOURCE_DIR / 'dist' / 'ds2md-viewer.zip'


def build_archive(output: Path) -> Path:
    output = output.expanduser().resolve()
    if output.suffix.lower() != '.zip':
        raise ValueError('The output filename must end in .zip.')
    for filename in RUNTIME_FILES:
        if not (SOURCE_DIR / filename).is_file():
            raise FileNotFoundError(f'Required viewer file is missing: {SOURCE_DIR / filename}')
    output.parent.mkdir(parents=True, exist_ok=True)
    # Complete the ZIP before replacing an earlier build; no recursive globbing.
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
    parser = argparse.ArgumentParser(description='Pack viewer Python sources and requirements only; no game resources.')
    parser.add_argument('-o', '--output', type=Path, default=DEFAULT_OUTPUT,
                        help='ZIP filename (default: ds2_research/dist/ds2md-viewer.zip)')
    args = parser.parse_args()
    try:
        output = build_archive(args.output)
    except (OSError, ValueError) as error:
        parser.exit(1, f'Cannot pack viewer: {error}\n')
    print(f'Created: {output}')
    print(f'Archive size: {output.stat().st_size} bytes')
    print('Included:')
    for filename in RUNTIME_FILES:
        print(f'  {ARCHIVE_DIRECTORY}/{filename}')
    print('No models, textures or other game resources included.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
