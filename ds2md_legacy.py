"""Decode observed DS2MD text 0.5 and binary 0.6 layouts.

Geometry is normalized to flat tuples; sampled matrices remain in disk order.
The binary entry point accepts inspect_ds2md.Reader after its version string,
without importing that module. Unknown layout branches are deliberately rejected.
"""

from math import isfinite
import re


_MAGIC = 'DS2ModelFile_1'
_BINARY_HEADER = b'\x0e\x00DS2ModelFile_1\x03\x000.6'
_IDENTITY = (
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
)
_WHITESPACE = b' \t\r\n\v\f'
_SPACE_PATTERN = re.compile(rb'[ \t\r\n\v\f]*')
_TOKEN_PATTERN = re.compile(rb'[^ \t\r\n\v\f]+')
_INTEGER_PATTERN = re.compile(rb'[+-]?[0-9]+')
_REAL_PATTERN = re.compile(
    rb'[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?'
)


class _TextReader:
    """Read one token at a time, preserving literal backslashes in strings."""

    def __init__(self, data):
        self.data = data
        self.pos = 0

    def whitespace(self):
        self.pos = _SPACE_PATTERN.match(self.data, self.pos).end()

    def token(self, quoted):
        self.whitespace()
        start = self.pos
        if start == len(self.data):
            raise ValueError(f'EOF at 0x{start:X}: expected text token')
        if quoted:
            if self.data[start] != 34:
                raise ValueError(f'Expected quoted string at 0x{start:X}')
            end = self.data.find(b'"', start + 1)
            if end < 0:
                raise ValueError(f'Unterminated quoted string at 0x{start:X}')
            self.pos = end + 1
            value = self.data[start + 1:end]
        else:
            match = _TOKEN_PATTERN.match(self.data, start)
            self.pos = match.end()
            value = match.group()
        if self.pos < len(self.data) and self.data[self.pos] not in _WHITESPACE:
            raise ValueError(f'Missing token separator at 0x{self.pos:X}')
        return start, value

    def string(self):
        _, value = self.token(True)
        return value.decode('cp1251', errors='replace')

    def integer(self):
        start, value = self.token(False)
        if not _INTEGER_PATTERN.fullmatch(value):
            raise ValueError(f'Invalid integer token at 0x{start:X}')
        try:
            return int(value)
        except ValueError as error:
            raise ValueError(f'Invalid integer token at 0x{start:X}') from error

    def count(self, label):
        start = self.pos
        value = self.integer()
        if value < 0:
            raise ValueError(f'Negative {label} at 0x{start:X}')
        return value

    def real(self):
        start, value = self.token(False)
        if not _REAL_PATTERN.fullmatch(value):
            raise ValueError(f'Invalid real token at 0x{start:X}')
        result = float(value)
        if not isfinite(result):
            raise ValueError(f'Non-finite real token at 0x{start:X}')
        return result

    def array(self, count, integer=False):
        # Every token needs at least one byte. Check before count-driven work;
        # do not materialize a token list for the whole file.
        if count < 0 or count > len(self.data) - self.pos:
            raise ValueError(f'EOF at 0x{self.pos:X}: need {count} text values')
        read_value = self.integer if integer else self.real
        return tuple(read_value() for _ in range(count))


def _zero(value, label, offset):
    if value != 0:
        raise ValueError(f'Unsupported {label} {value} at 0x{offset:X}')


def _rate(value, offset):
    if not isfinite(value) or value <= 0.0:
        raise ValueError(f'Invalid sample rate {value} at 0x{offset:X}')
    return value


def _finite(values, label, offset):
    if any(not isfinite(value) for value in values):
        raise ValueError(f'Non-finite {label} at 0x{offset:X}')
    return values


def _geometry(mesh, vertices, indices):
    offset = mesh['offset']
    if mesh['index_count'] != 3 * mesh['primitive_count']:
        raise ValueError(f'Inconsistent triangle count at 0x{offset:X}')
    if any(index < 0 or index >= mesh['vertex_count'] for index in indices):
        raise ValueError(f'Index outside vertex array at 0x{offset:X}')
    _finite(vertices, 'vertex data', mesh['vertices_offset'])
    mesh['positions'] = tuple(
        vertices[start + axis]
        for start in range(0, len(vertices), 8) for axis in range(3)
    )
    mesh['normals'] = tuple(
        vertices[start + axis]
        for start in range(0, len(vertices), 8) for axis in range(3, 6)
    )
    mesh['uv_coords'] = tuple(
        vertices[start + axis]
        for start in range(0, len(vertices), 8) for axis in range(6, 8)
    )
    mesh['indices'] = indices
    mesh['triangle_strip'] = 0
    mesh['matrix_tracks'] = []
    mesh['vertex_influences'] = ()


def _bounds(mesh, values):
    _finite(values, 'bounds', mesh['bounds_offset'])
    for start in range(0, len(values), 6):
        if any(values[start + axis] > values[start + axis + 3] for axis in range(3)):
            raise ValueError(f'Inverted bounds at 0x{mesh["bounds_offset"]:X}')
    mesh['bounds'] = values
    mesh['bounds_min'] = values[:3]
    mesh['bounds_max'] = values[3:6]


def parse_text_model(data: bytes, name: str) -> dict:
    """Parse quoted text 0.5, including static skies and rigid sampled meshes."""
    reader = _TextReader(data)
    signature = reader.string()
    if signature != _MAGIC:
        raise ValueError('Invalid DS2ModelFile_1 signature')
    version = reader.string()
    if version != '0.5':
        raise ValueError(f'Unsupported text DS2MD version {version!r}')
    out = {
        'version': version, 'name': name, 'signature': signature,
        'meshes': [], 'has_skeleton': 0, 'collision_objects': [],
    }
    mesh_count = reader.count('mesh count')
    out['mesh_count'] = mesh_count
    for _ in range(mesh_count):
        reader.whitespace()
        mesh = {
            'offset': reader.pos, 'name': reader.string(),
            'material': reader.string(),
            'textures': [reader.string() for _ in range(5)],
        }
        mesh['counts_offset'] = reader.pos
        mesh['primitive_count'] = reader.count('triangle count')
        mesh['vertex_count'] = reader.count('vertex count')
        mesh['index_count'] = reader.count('index count')
        field_offset = reader.pos
        mesh['unknown_A'] = reader.integer()
        _zero(mesh['unknown_A'], 'text unknown_A', field_offset)
        mesh['vertices_offset'] = reader.pos
        vertices = reader.array(mesh['vertex_count'] * 8)
        mesh['indices_offset'] = reader.pos
        indices = reader.array(mesh['index_count'], integer=True)
        _geometry(mesh, vertices, indices)
        mesh['sample_count_offset'] = reader.pos
        mesh['sample_count'] = reader.count('sample count')
        field_offset = reader.pos
        mesh['sample_rate'] = _rate(reader.real(), field_offset)
        mesh['bounds_offset'] = reader.pos
        _bounds(mesh, reader.array(max(1, mesh['sample_count']) * 6))
        field_offset = reader.pos
        mesh['unknown_B'] = reader.integer()
        _zero(mesh['unknown_B'], 'text unknown_B', field_offset)
        field_offset = reader.pos
        mesh['has_matrices'] = reader.integer()
        if mesh['has_matrices'] not in (0, 1):
            raise ValueError(
                f'Unsupported text has_matrices {mesh["has_matrices"]} '
                f'at 0x{field_offset:X}'
            )
        if bool(mesh['sample_count']) != bool(mesh['has_matrices']):
            raise ValueError(f'Unsupported text sample/matrix layout at 0x{field_offset:X}')
        mesh['legacy_mode'] = 2 if mesh['has_matrices'] else 3
        if mesh['has_matrices']:
            mesh['matrices_offset'] = reader.pos
            mesh['matrix_tracks'].append({
                'pre_matrix': _IDENTITY,
                'samples': reader.array(mesh['sample_count'] * 16),
            })
        field_offset = reader.pos
        mesh['unknown_C'] = reader.integer()
        _zero(mesh['unknown_C'], 'text unknown_C', field_offset)
        mesh['end'] = reader.pos
        out['meshes'].append(mesh)
    out['terminal_offset'] = reader.pos
    out['terminal'] = reader.integer()
    _zero(out['terminal'], 'text terminal', out['terminal_offset'])
    reader.whitespace()
    if reader.pos != len(data):
        raise ValueError(f'{len(data) - reader.pos} trailing bytes at 0x{reader.pos:X}')
    out['end'] = reader.pos
    out['file_size'] = len(data)
    return out


def parse_binary_model(reader, name: str) -> dict:
    """Parse binary 0.6 with Reader positioned just after magic and version.

    Mode 1 stores track-major pre/sample matrices and per-vertex track weights;
    mode 2 stores one rigid track; mode 3 is static. Legacy collision type 1
    has opaque bytes rather than a modern bone index. No skeleton byte follows.
    """
    if reader.pos != len(_BINARY_HEADER) or not reader.data.startswith(_BINARY_HEADER):
        raise ValueError('Expected binary DS2MD 0.6 reader after magic/version at 0x15')
    header_offset = reader.pos
    sample_rate = _rate(reader.value('f'), header_offset)
    out = {
        'version': '0.6', 'name': name, 'signature': _MAGIC,
        'header_parameter': sample_rate, 'sample_rate': sample_rate,
        'meshes': [], 'has_skeleton': 0, 'collision_objects': [],
    }
    mesh_count = reader.value('H')
    out['mesh_count'] = mesh_count
    for _ in range(mesh_count):
        mesh = {
            'offset': reader.pos, 'name': reader.string(),
            'material': reader.string(),
            'textures': [reader.string() for _ in range(5)],
        }
        mesh['counts_offset'] = reader.pos
        mesh['primitive_count'] = reader.value('I')
        mesh['vertex_count'] = reader.value('I')
        mesh['index_count'] = reader.value('I')
        field_offset = reader.pos
        mesh['unknown_before_vertices'] = reader.value('B')
        _zero(mesh['unknown_before_vertices'], 'pre-vertex byte', field_offset)
        mesh['vertices_offset'] = reader.pos
        vertices = reader.array('f', mesh['vertex_count'] * 8)
        mesh['indices_offset'] = reader.pos
        indices = reader.array('H', mesh['index_count'])
        _geometry(mesh, vertices, indices)
        mesh['sample_count_offset'] = reader.pos
        samples = mesh['sample_count'] = reader.value('I')
        field_offset = reader.pos
        mode = mesh['legacy_mode'] = reader.value('B')
        mesh['sample_rate'] = sample_rate
        if mode not in (1, 2, 3):
            raise ValueError(f'Unsupported legacy animation mode {mode} at 0x{field_offset:X}')
        if (mode == 3 and samples != 0) or (mode != 3 and samples == 0):
            raise ValueError(f'Unsupported legacy mode/sample layout at 0x{field_offset:X}')
        mesh['bounds_offset'] = reader.pos
        _bounds(mesh, reader.array('f', max(1, samples) * 6))
        if mode == 1:
            mesh['track_count_offset'] = reader.pos
            track_count = mesh['track_count'] = reader.value('H')
            if not track_count:
                raise ValueError(f'Empty sampled track palette at 0x{mesh["track_count_offset"]:X}')
            mesh['matrices_offset'] = reader.pos
            for _ in range(track_count):
                offset = reader.pos
                pre_matrix = _finite(reader.array('f', 16), 'pre-matrix', offset)
                offset = reader.pos
                track_samples = _finite(
                    reader.array('f', samples * 16), 'sample matrices', offset
                )
                mesh['matrix_tracks'].append({
                    'pre_matrix': pre_matrix, 'samples': track_samples,
                })
            mesh['assignment_count_offset'] = reader.pos
            assignment_count = mesh['assignment_count'] = reader.value('I')
            if assignment_count != mesh['vertex_count']:
                raise ValueError(
                    f'Assignment count differs from vertex count '
                    f'at 0x{mesh["assignment_count_offset"]:X}'
                )
            mesh['assignments_offset'] = reader.pos
            assignments = []
            influence_counts = {}
            for _ in range(assignment_count):
                count = reader.value('B')
                influence_counts[count] = influence_counts.get(count, 0) + 1
                influences = []
                for _ in range(count):
                    offset = reader.pos
                    track, weight = reader.value('Hf')
                    if track >= track_count:
                        raise ValueError(f'Track index outside palette at 0x{offset:X}')
                    if not isfinite(weight) or weight < 0.0:
                        raise ValueError(f'Invalid track weight at 0x{offset:X}')
                    influences.append((track, weight))
                assignments.append(tuple(influences))
            mesh['vertex_influences'] = tuple(assignments)
            mesh['influence_counts'] = influence_counts
        elif mode == 2:
            mesh['matrices_offset'] = reader.pos
            mesh['matrix_tracks'].append({
                'pre_matrix': _IDENTITY,
                'samples': _finite(
                    reader.array('f', samples * 16), 'sample matrices', mesh['matrices_offset']
                ),
            })
        mesh['end'] = reader.pos
        out['meshes'].append(mesh)
    out['collision_count_offset'] = reader.pos
    collision_count = reader.value('H')
    out['collision_count'] = collision_count
    for _ in range(collision_count):
        obj = {'offset': reader.pos, 'name': reader.string()}
        field_offset = reader.pos
        obj['type'] = reader.value('B')
        if obj['type'] != 1:
            raise ValueError(f'Unsupported legacy collision type {obj["type"]} at 0x{field_offset:X}')
        obj['opaque_offset'] = reader.pos
        start = reader.take(8)
        obj['opaque'] = reader.data[start:start + 8].hex()
        obj['parameters_offset'] = reader.pos
        obj['parameters'] = _finite(
            reader.array('f', 3), 'collision parameters', obj['parameters_offset']
        )
        obj['matrix_offset'] = reader.pos
        obj['matrix'] = _finite(reader.array('f', 16), 'collision matrix', obj['matrix_offset'])
        obj['end'] = reader.pos
        out['collision_objects'].append(obj)
    if reader.pos != len(reader.data):
        raise ValueError(f'{len(reader.data) - reader.pos} trailing bytes at 0x{reader.pos:X}')
    out['end'] = reader.pos
    out['file_size'] = len(reader.data)
    return out
