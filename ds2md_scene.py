"""Geometry, skeletal data and local albedo references from binary DS2MD files.

Vertex arrays remain in their stored bind space. The renderer applies the
animated global bone matrix times inverse global bind, as established by
ds2render.dll:0x100CD657 and 0x100D7C00 (preferred base 0x10000000).
Bounds here describe the stored vertices; the viewer evaluates animated bounds.
"""

from dataclasses import dataclass
from math import isfinite
from pathlib import Path, PureWindowsPath
import struct

if __package__:
    from .inspect_ds2md import UnsupportedVersion, parse_model
    from .ds2md_animation import Skeleton, load_skeleton
else:
    from inspect_ds2md import UnsupportedVersion, parse_model
    from ds2md_animation import Skeleton, load_skeleton


# A viewer safety policy, not a restriction claimed for the DS2MD format.
# Reject corrupt-but-finite floats before they reach camera and GPU calculations.
_MAX_COORDINATE = 1_000_000_000.0


@dataclass(frozen=True, slots=True)
class Mesh:
    name: str
    positions: tuple[float, ...]
    normals: tuple[float, ...]
    indices: tuple[int, ...]
    uv_coords: tuple[float, ...]
    albedo_path: Path | None
    albedo_reference: str
    influence_ranges: tuple[int, ...]
    influences: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class Scene:
    path: Path
    name: str
    version: str
    meshes: tuple[Mesh, ...]
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]
    note: str
    texture_warnings: tuple[str, ...]
    skeleton: Skeleton | None


def _strip_triangles(
    indices: tuple[int, ...], positions: tuple[float, ...]
) -> tuple[int, ...]:
    """Expand GL_TRIANGLE_STRIP, retaining parity across degenerate joins."""
    triangles = []
    for step in range(len(indices) - 2):
        a, b, c = indices[step:step + 3]
        if step & 1:
            a, b = b, a
        if a == b or b == c or c == a:
            continue
        # Different indices can still describe the same point or a straight line.
        ai, bi, ci = a * 3, b * 3, c * 3
        ux = positions[bi] - positions[ai]
        uy = positions[bi + 1] - positions[ai + 1]
        uz = positions[bi + 2] - positions[ai + 2]
        vx = positions[ci] - positions[ai]
        vy = positions[ci + 1] - positions[ai + 1]
        vz = positions[ci + 2] - positions[ai + 2]
        if (uy * vz - uz * vy == 0.0
                and uz * vx - ux * vz == 0.0
                and ux * vy - uy * vx == 0.0):
            continue
        triangles.extend((a, b, c))
    return tuple(triangles)


def _texture_index() -> dict[str, list[Path]]:
    """Index only the game's local texture tree, never serialized source paths."""
    root = Path(__file__).resolve().parent.parent / 'gameres' / 'textures'
    images: dict[str, list[Path]] = {}
    for path in root.rglob('*'):
        if path.suffix.casefold() in ('.dds', '.tga') and path.is_file():
            images.setdefault(path.name.casefold(), []).append(path)
    return images


def _resolve_albedo(
    reference: str, images: dict[str, list[Path]], mesh_name: str
) -> tuple[Path | None, str | None]:
    # PureWindowsPath interprets drive/UNC references without accessing them.
    stem = PureWindowsPath(reference).stem.casefold()
    for extension in ('.dds', '.tga'):
        candidates = images.get(stem + extension, ())
        if len(candidates) == 1:
            return candidates[0], None
        if candidates:
            matches = ', '.join(str(path) for path in sorted(candidates))
            return None, (
                f"Mesh {mesh_name}: ambiguous albedo reference {reference!r}: "
                f"{matches}. Keeping the material untextured."
            )
    return None, (
        f"Mesh {mesh_name}: no local DDS/TGA for albedo reference {reference!r}. "
        "Keeping the material untextured."
    )


def load_scene(path: Path | str) -> Scene:
    """Load stored geometry and local albedo references; raise ValueError/OSError.

    The existing metadata parser remains the sole authority for layout/version
    decoding. Array offsets refer to the original bytes, including for strips.
    Triangle lists preserve their original index order and triangle membership.
    """
    path = Path(path)
    try:
        metadata = parse_model(path)
    except UnsupportedVersion as error:
        raise ValueError(
            f"Unsupported DS2MD version {error}. "
            "This viewer supports standard binary 0.9 and 1.0 only."
        ) from error
    data = path.read_bytes()
    if len(data) != metadata['file_size']:
        raise ValueError("The model changed while it was being opened; open it again.")

    skeleton = load_skeleton(metadata, data)
    bone_count = len(skeleton.bones) if skeleton is not None else 0
    meshes = []
    images = _texture_index()
    texture_warnings = []
    lower = [float('inf')] * 3
    upper = [float('-inf')] * 3
    triangle_count = 0
    for number, source in enumerate(metadata['meshes'], 1):
        name = f"{number:02d}: {source['material'] or '(unnamed material)'}"
        count = source['vertex_count']
        positions = struct.unpack_from(f'<{count * 3}f', data, source['positions_offset'])
        normals = struct.unpack_from(f'<{count * 3}f', data, source['normals_offset'])
        uv_coords = struct.unpack_from(f'<{count * 2}f', data, source['uv_offset'])
        indices = struct.unpack_from(
            f"<{source['index_count']}H", data, source['indices_offset']
        )
        if any(not isfinite(value) for value in positions):
            raise ValueError(f"Mesh {name} contains non-finite vertex positions.")
        if any(abs(value) > _MAX_COORDINATE for value in positions):
            raise ValueError(
                f"Mesh {name} exceeds the viewer's coordinate safety range "
                f"(+/-{_MAX_COORDINATE:g} units)."
            )
        if any(not isfinite(value) for value in normals):
            raise ValueError(f"Mesh {name} contains non-finite vertex normals.")
        ranges = []
        influences = []
        assignment_offset = source['assignments_offset']
        for _ in range(count):
            start = len(influences) // 2
            influence_count = 0
            if source['influence_mode']:
                influence_count = data[assignment_offset]
                assignment_offset += 1
                if influence_count >= 10:
                    raise ValueError(f"Mesh {name} has more than nine influences per vertex.")
                for _ in range(influence_count):
                    bone, weight = struct.unpack_from('<Hf', data, assignment_offset)
                    assignment_offset += 6
                    if bone >= bone_count:
                        raise ValueError(f"Mesh {name} refers to missing bone {bone}.")
                    if not isfinite(weight) or weight < 0.0:
                        raise ValueError(f"Mesh {name} contains an invalid bone weight.")
                    influences.extend((float(bone), weight))
            ranges.extend((start, influence_count))
        if assignment_offset != source['end']:
            raise ValueError(f"Mesh {name} has inconsistent influence data.")
        # Check against these bytes too: the metadata parser opens the path itself.
        if indices and max(indices) >= count:
            raise ValueError(f"Mesh {name} has an index outside its vertex array.")
        if source['triangle_strip']:
            indices = _strip_triangles(indices, positions)
        elif len(indices) % 3:
            raise ValueError(f"Mesh {name} has an incomplete triangle list.")
        triangle_count += len(indices) // 3
        for offset in range(0, len(positions), 3):
            for axis in range(3):
                value = positions[offset + axis]
                lower[axis] = min(lower[axis], value)
                upper[axis] = max(upper[axis], value)
        reference = source['textures'][0]
        albedo_path, warning = _resolve_albedo(reference, images, name)
        if warning:
            texture_warnings.append(warning)
        meshes.append(
            Mesh(name, positions, normals, indices, uv_coords, albedo_path, reference,
                 tuple(ranges), tuple(influences))
        )

    if not triangle_count:
        raise ValueError("The model contains no renderable triangles.")
    note = "Bind-space geometry; camera fits the displayed pose."
    if skeleton is not None:
        note += f" {len(skeleton.bones)} bones; {len(skeleton.clips)} animation clips."
    return Scene(
        path=path,
        name=metadata['name'] or path.stem,
        version=metadata['version'],
        meshes=tuple(meshes),
        bounds_min=tuple(lower),
        bounds_max=tuple(upper),
        note=note,
        texture_warnings=tuple(texture_warnings),
        skeleton=skeleton,
    )
