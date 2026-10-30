"""Inspect DS2GeometryFile_1 0.8 maps, without third-party dependencies.

Run: python ds2_research/inspect_ds2.py FILE_OR_DIRECTORY [--meshes]
Offsets point into the original bytes. Unknown semantics retain memory-field
names from ds2render.dll instead of speculative labels. See DS2_RESEARCH.md.
"""

from collections import Counter
from pathlib import Path
import argparse
import json
import math
import struct

from inspect_ds2md import Reader, UnsupportedVersion


# Disk order is NOT numerical bit order. Colors are four u32 channels on disk.
ATTRIBUTES = ((2, 'positions', 12), (8, 'uv', 8), (16, 'lightmap_uv', 8),
              (32, 'colors', 16), (4, 'normals', 12),
              (64, 'tangents', 12), (128, 'binormals', 12))


class MapReader(Reader):
    def __init__(self, path, data=None):
        self.data = Path(path).read_bytes() if data is None else data
        self.pos = 0

    def expect(self, text):
        start = self.pos
        actual = self.string()
        if actual != text:
            raise ValueError(f'At 0x{start:X}: expected {text!r}, got {actual!r}')

    def count(self, minimum_size):
        count = self.value('I')
        if count > (len(self.data) - self.pos) // minimum_size:
            raise ValueError(f'Impossible count {count} at 0x{self.pos - 4:X}')
        return count


def _mesh(r):
    m = {'offset': r.pos, 'instanced': r.value('B'), 'material': r.string(),
         'textures': [r.string() for _ in range(5)]}
    m['bounds_min'] = r.array('f', 3)
    m['bounds_max'] = r.array('f', 3)
    m['sphere_center'] = r.array('f', 3)
    m['sphere_radius'] = r.value('f')
    m['obb_offset'] = r.take(156)
    m['flag_114'] = r.value('B')
    m['vector_118'] = r.array('f', 3)
    fields = r.array('I', 12)
    m['raw_draw_fields'] = fields
    names = ('lightmap_index', 'primitive_count', 'vertex_count', 'field_d8',
             'index_count', 'index_buffer', 'index_start', 'strip_index_count',
             'strip_index_buffer', 'strip_index_start', 'vertex_buffer', 'vertex_start')
    m.update(zip(names, fields))
    if m['lightmap_index'] == 0xFFFFFFFF:
        m['lightmap_index'] = -1
    if m['instanced']:
        m['instance_fields'] = r.array('I', 2)
        m['matrix'] = r.array('f', 16)
        m['instance_tail'] = r.array('I', 2)
    m['end'] = r.pos
    return m


def _buffer(r):
    out = {'offset': r.pos, 'flags': r.value('I')}
    flags = out['flags']
    if flags & ~255 or not flags:
        raise ValueError(f'Unknown buffer flags 0x{flags:X} at 0x{out["offset"]:X}')
    out['count'] = r.value('I')
    if flags & 1:
        out['indices_offset'] = r.take(out['count'] * 4)
    else:
        for bit, name, stride in ATTRIBUTES:
            if flags & bit:
                out[name + '_offset'] = r.take(out['count'] * stride)
    out['end'] = r.pos
    return out


def _node(r):
    out = {'offset': r.pos, 'bounds_min': r.array('f', 3),
           'bounds_max': r.array('f', 3), 'field_24': r.value('I'),
           'center': r.array('f', 3), 'leaf': r.value('B')}
    if out['leaf']:
        out['mesh_indices'] = r.array('I', r.count(4))
    else:
        out['children'] = r.array('I', 8)
    out['end'] = r.pos
    return out


def _model(r):
    out = {'offset': r.pos, 'name': r.string(), 'bounds_min': r.array('f', 3),
           'bounds_max': r.array('f', 3), 'instances': []}
    for _ in range(r.count(88)):
        out['instances'].append({'matrix': r.array('f', 16),
                                 'bounds_min': r.array('f', 3),
                                 'bounds_max': r.array('f', 3)})
    out['end'] = r.pos
    return out


def parse_map(path, data=None):
    """Decode every chunk sequentially; reject truncation and unknown chunks."""
    r = MapReader(path, data)
    r.expect('DS2GeometryFile_1')
    version = r.string()
    if version != '0.8':
        raise UnsupportedVersion(f'DS2 geometry {version}')
    out = {'version': version, 'file_size': len(r.data), 'chunks': [],
           'meshes': [], 'buffers': [], 'lightmaps': [], 'vistree': [],
           'lights': [], 'models': []}
    seen = set()
    while r.pos < len(r.data):
        start = r.pos
        r.expect('DS2Chunk_begin')
        name = r.string()
        if name in seen:
            raise ValueError(f'Duplicate chunk {name!r} at 0x{start:X}')
        seen.add(name)
        chunk = {'name': name, 'offset': start, 'payload_offset': r.pos}
        if name == 'meshes':
            out[name] = [_mesh(r) for _ in range(r.count(270))]
        elif name == 'buffers':
            out[name] = [_buffer(r) for _ in range(r.count(8))]
        elif name == 'lightmaps':
            out[name] = [r.string() for _ in range(r.count(2))]
        elif name == 'vistree':
            out[name] = [_node(r) for _ in range(r.count(45))]
        elif name == 'lights':
            for _ in range(r.count(72)):
                out[name].append({'offset': r.pos, 'type': r.value('I'),
                                  'vectors': r.array('f', 9),
                                  'parameters': r.array('I', 8)})
        elif name == 'models':
            out[name] = [_model(r) for _ in range(r.count(30))]
        else:
            raise ValueError(f'Unknown chunk {name!r} at 0x{start:X}; no size to safely skip it')
        chunk['count'] = len(out[name])
        chunk['payload_end'] = r.pos
        r.expect('DS2Chunk_end')
        chunk['end'] = r.pos
        out['chunks'].append(chunk)
    if not {'meshes', 'buffers', 'lightmaps'}.issubset(seen):
        raise ValueError('Missing geometry, buffers or lightmaps chunk')
    _validate(out, r.data)
    out['end'] = r.pos
    return out


def _validate(out, data):
    buffers = out['buffers']
    for number, m in enumerate(out['meshes']):
        def fail(message):
            raise ValueError(f'Mesh {number} at 0x{m["offset"]:X}: {message}')
        if any(not math.isfinite(v) for v in (*m['bounds_min'], *m['bounds_max'])):
            fail('non-finite bounds')
        if m['vertex_buffer'] >= len(buffers):
            fail('vertex_buffer outside buffers')
        vb = buffers[m['vertex_buffer']]
        if 'positions_offset' not in vb:
            fail('vertex buffer has no positions')
        if m['vertex_start'] + m['vertex_count'] > vb['count']:
            fail('vertex range outside buffer')
        for prefix in ('', 'strip_'):
            count = m[prefix + 'index_count']
            if not count:
                continue
            index = m[prefix + 'index_buffer']
            if index >= len(buffers) or 'indices_offset' not in buffers[index]:
                fail(f'{prefix}index_buffer outside index buffers')
            ib = buffers[index]
            start = m[prefix + 'index_start']
            if start + count > ib['count']:
                fail(f'{prefix}index range outside buffer')
            indices = struct.iter_unpack('<I', memoryview(data)[
                ib['indices_offset'] + start * 4:ib['indices_offset'] + (start + count) * 4])
            if any(value[0] + m['vertex_start'] >= vb['count'] for value in indices):
                fail(f'{prefix}index plus vertex base outside vertex buffer')
        if m['lightmap_index'] >= len(out['lightmaps']):
            fail('lightmap index outside table')


def summary(path, metadata):
    meshes = metadata['meshes']
    return {'path': str(path), 'version': metadata['version'],
            'file_size': metadata['file_size'], 'chunks': metadata['chunks'],
            'mesh_count': len(meshes),
            'primitive_count': sum(m['primitive_count'] for m in meshes),
            'vertex_buffer_count': sum(not b['flags'] & 1 for b in metadata['buffers']),
            'stored_vertices': sum(b['count'] for b in metadata['buffers'] if 'positions_offset' in b),
            'stored_indices': sum(b['count'] for b in metadata['buffers'] if b['flags'] & 1),
            'instanced_mesh_count': sum(bool(m['instanced']) for m in meshes),
            'materials': dict(Counter(m['material'] for m in meshes)),
            'lightmaps': metadata['lightmaps'],
            'bounds_min': tuple(min(m['bounds_min'][a] for m in meshes) for a in range(3)) if meshes else None,
            'bounds_max': tuple(max(m['bounds_max'][a] for m in meshes) for a in range(3)) if meshes else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path)
    parser.add_argument('--meshes', action='store_true', help='Include decoded records and array offsets')
    args = parser.parse_args()
    paths = sorted(args.path.rglob('*.ds2')) if args.path.is_dir() else [args.path]
    results, failures = [], []
    for path in paths:
        try:
            metadata = parse_map(path)
            result = summary(path, metadata)
            if args.meshes:
                result['records'] = metadata
            results.append(result)
        except (OSError, ValueError) as error:
            failures.append({'path': str(path), 'error': str(error)})
    if args.path.is_dir():
        report = {'map_count': len(results), 'maps': results, 'failures': failures}
        if not paths:
            failures.append({'path': str(args.path), 'error': 'No .ds2 files found'})
    elif failures:
        report = {'failures': failures}
    else:
        report = results[0]
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
