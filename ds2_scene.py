"""Static DS2 map scene, local textures and material declarations.

No game code executes. Geometry arrays remain in one immutable byte buffer;
shared geometry and per-instance color streams are not expanded on the CPU.
"""

from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
import math
import re
import struct

from inspect_ds2 import parse_map


@dataclass(frozen=True, slots=True)
class Material:
    name: str
    shader: str = 'unknown'
    static_light: str = ''
    vertex_colors: bool = False
    self_illumination: bool = False
    overbright: bool = True
    alpha_cutoff: float = -1.0
    blend: str = ''
    cull: bool = True
    depth_write: bool = True
    depth_test: bool = True
    depth_offset: bool = False
    sort: int = 5
    color: tuple = (1.0, 1.0, 1.0)
    diffuse: str = '$diffuse$'
    clamp: bool = False
    uv_transform: bool = False


@dataclass(frozen=True, slots=True)
class MapScene:
    path: Path
    data: bytes
    metadata: dict
    materials: dict
    albedos: dict
    lightmaps: tuple
    bounds_min: tuple
    bounds_max: tuple
    warnings: tuple


class TextureIndex:
    def __init__(self, root):
        self.root = Path(root)
        self.images = {}
        for path in sorted(self.root.rglob('*')):
            if path.is_file() and path.suffix.casefold() in ('.dds', '.tga'):
                self.images.setdefault(path.name.casefold(), []).append(path)

    def resolve(self, reference, nearby):
        """Prefer sibling images, then unique local DDS/TGA, never source drives/UNC."""
        source = PureWindowsPath(reference)
        name = source.name
        if not name or reference.casefold() == 'notexture':
            return None, f'No texture for {reference!r}'
        names = [source.stem + extension for extension in ('.dds', '.tga')]
        for candidate in names:
            siblings = [p for p in nearby.iterdir() if p.name.casefold() == candidate.casefold()]
            if len(siblings) == 1:
                return siblings[0], None
        for candidate in names:
            matches = self.images.get(candidate.casefold(), [])
            if len(matches) == 1:
                return matches[0], None
            if len(matches) > 1:
                # Preserve the reference's texture-tree suffix where possible.
                parts = tuple(p.casefold() for p in source.parts)
                exact = [p for p in matches if tuple(x.casefold() for x in p.parts[-2:-1]) == parts[-2:-1]]
                if len(exact) == 1:
                    return exact[0], None
                return None, f'Ambiguous local texture {reference!r}: {len(matches)} candidates'
        return None, f'Missing local DDS/TGA for {reference!r}'


def _fields(tokens):
    """Return only depth-one fields; nested texture/compiler/animation data stay grouped."""
    fields = {}
    i = 0
    while i + 1 < len(tokens):
        name, value = tokens[i:i + 2]
        i += 2
        if value == '{':
            depth, group = 1, []
            while i < len(tokens) and depth:
                value = tokens[i]
                i += 1
                depth += (value == '{') - (value == '}')
                if depth:
                    group.append(value)
            fields[name.casefold()] = group
        else:
            fields[name.casefold()] = [value]
    return fields


def read_materials(directory):
    materials = {}
    for path in sorted(Path(directory).glob('*.mat')):
        text = path.read_text(encoding='cp1251')
        # An unterminated block comment in shipped ds2_man.mat extends to EOF.
        text = re.sub(r'/\*.*?(?:\*/|\Z)|//[^\r\n]*', '', text, flags=re.S)
        tokens = re.findall(r'"[^"]*"|[{}]|[^\s{}]+', text)
        i = 0
        while i + 2 < len(tokens):
            if tokens[i].casefold() != 'template_material' or tokens[i + 2] != '{':
                i += 1
                continue
            name = tokens[i + 1]
            i += 3
            start, depth = i, 1
            while i < len(tokens) and depth:
                depth += (tokens[i] == '{') - (tokens[i] == '}')
                i += 1
            fields = _fields(tokens[start:i - 1])
            def scalar(key, default=''):
                return fields.get(key, [default])[0].strip('"').casefold()
            alpha = fields.get('alphatest', [])
            color = fields.get('color3', ['1', '1', '1'])
            diffuse = fields.get('diffuse_texture', ['$white$'])
            materials[name.casefold()] = Material(
                name=name, shader=scalar('material', 'unknown'),
                static_light=scalar('static_light'),
                vertex_colors=scalar('use_vertex_colors') == 'true' or scalar('color') == 'vertex',
                self_illumination=scalar('self_illumination') == 'true',
                overbright=scalar('overbright', 'true') != 'false',
                alpha_cutoff=float(alpha[1]) if len(alpha) >= 2 and alpha[0] == 'greater' else -1.0,
                blend=scalar('blend'), cull=scalar('cull_face') != 'disable',
                depth_write=scalar('depth_write') not in ('false', 'disable'),
                depth_test=scalar('depth_test') != 'disable',
                depth_offset=scalar('depth_offset') == 'true',
                sort=int(scalar('sort_value', '5')), color=tuple(map(float, color[:3])),
                diffuse=diffuse[0].strip('"'), clamp='clampmap' in diffuse,
                uv_transform='diffuse_texture_transform' in fields)
    return materials


def load_map_scene(path, texture_index=None, materials=None):
    path = Path(path).resolve()
    data = path.read_bytes()
    metadata = parse_map(path, data)
    # A map copied elsewhere can still use the adjacent game's resources.
    game = next((p.parent for p in path.parents if p.name.casefold() == 'gameres'),
                Path(__file__).resolve().parent.parent)
    texture_index = texture_index or TextureIndex(game / 'gameres' / 'textures')
    materials = materials if materials is not None else read_materials(game / 'gameres' / 'materials')
    warnings, albedos = [], {}
    for buffer in metadata['buffers']:
        # Only positions affect camera safety. This renderer never consumes
        # normals/tangents; some shipped unused vectors contain NaNs. Finite
        # exporter UVs can be enormous and must not inherit a position limit.
        for name, stride in (('positions', 12), ('uv', 8), ('lightmap_uv', 8)):
            if name + '_offset' not in buffer:
                continue
            start = buffer[name + '_offset']
            values = struct.iter_unpack('<f', memoryview(data)[start:start + buffer['count'] * stride])
            if any(not math.isfinite(v[0]) or (name == 'positions' and abs(v[0]) > 1e9) for v in values):
                raise ValueError(f'Unsafe {name} coordinates in buffer at 0x{buffer["offset"]:X}')
    for mesh in metadata['meshes']:
        name = mesh['material'].casefold()
        material = materials.get(name)
        if material is None:
            material = Material(mesh['material'])
            warnings.append(f'Undefined material {mesh["material"]!r}: showing static texture only')
            materials[name] = material
        reference = mesh['textures'][0] if material.diffuse == '$diffuse$' else material.diffuse
        if reference not in albedos:
            if reference in ('$black$', '$white$'):
                albedos[reference] = reference
            else:
                albedos[reference], warning = texture_index.resolve(reference, path.parent)
                if warning:
                    warnings.append(warning)
        if mesh['instanced']:
            if any(not math.isfinite(v) or abs(v) > 1e9 for v in mesh['matrix']):
                raise ValueError('Unsafe instance matrix')
            color_buffer, color_start = mesh['instance_fields']
            if color_buffer != 0xFFFFFFFF:
                if color_buffer >= len(metadata['buffers']):
                    raise ValueError('Instance color buffer outside buffer table')
                colors = metadata['buffers'][color_buffer]
                if 'colors_offset' not in colors or color_start + mesh['vertex_count'] > colors['count']:
                    raise ValueError('Instance color range outside color buffer')
        if material.shader in ('envmap', 'embm', 'space_distortion', 'flare') or 'refl' in name or name.startswith('cubeman'):
            warnings.append(f'{mesh["material"]}: static albedo/lighting only; reflection/distortion/flare effects not reproduced')
        if material.uv_transform:
            warnings.append(f'{mesh["material"]}: stored UVs shown without material UV animation/transform')
    lightmaps = []
    for reference in metadata['lightmaps']:
        resolved, warning = texture_index.resolve(reference, path.parent)
        lightmaps.append(resolved)
        if warning:
            warnings.append(warning)
    meshes = metadata['meshes']
    if not meshes:
        raise ValueError('Map contains no geometry')
    low = tuple(min(m['bounds_min'][a] for m in meshes) for a in range(3))
    high = tuple(max(m['bounds_max'][a] for m in meshes) for a in range(3))
    return MapScene(path, data, metadata, materials, albedos, tuple(lightmaps),
                    low, high, tuple(dict.fromkeys(warnings)))
