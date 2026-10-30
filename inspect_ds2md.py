"""Inspect DS2MD text 0.5 and binary 0.6/0.9/1.0; no third-party dependencies.

Usage: python ds2_research/inspect_ds2md.py FILE_OR_DIRECTORY
A file emits decoded metadata and byte offsets as JSON. A directory recursively
checks .ds2md files; failures exit with 1. DSMO packed/cache is not supported.
The report omits decoded geometry and sampled matrix arrays.
Strings are displayed as CP1251; the file itself contains untagged byte strings.

Evidence: ds2render.dll SHA256
216c22891726e624787918396112127f157de0255e70994be01fe509a3624c78
Preferred base 0x10000000. Main 0x100BC320; mesh 0x100C2D60;
collision 0x100B6EA0; skeleton 0x100CD2C0; animation 0x100D1370.
Matrix floats are in disk order; the engine transposes them on load.
The second animation float is time-valued, but its purpose is unconfirmed.
"""

from pathlib import Path
import struct
from collections import Counter

if __package__:
    from .ds2md_legacy import parse_binary_model, parse_text_model
else:
    from ds2md_legacy import parse_binary_model, parse_text_model

class UnsupportedVersion(ValueError):
    pass

class Reader:
    def __init__(self, path):
        self.data = Path(path).read_bytes()
        self.pos = 0
    def take(self, size):
        start = self.pos
        if size < 0 or start + size > len(self.data):
            raise ValueError(f'EOF at 0x{start:X}: need {size}, file size {len(self.data)}')
        self.pos += size
        return start
    def value(self, fmt):
        start = self.take(struct.calcsize('<' + fmt))
        values = struct.unpack_from('<' + fmt, self.data, start)
        return values[0] if len(values) == 1 else values
    def array(self, fmt, count):
        start = self.take(struct.calcsize('<' + fmt) * count)
        return struct.unpack_from('<' + str(count) + fmt, self.data, start)
    def string(self):
        size = self.value('H')
        start = self.take(size)
        return self.data[start:start + size].decode('cp1251', errors='replace')
    def indices(self):
        return self.array('H', self.value('H'))

def parse_model(path):
    r = Reader(path)
    if r.data.startswith(b'DSMO'):
        raise UnsupportedVersion('DSMO packed/cache')
    if r.data.lstrip(b' \t\r\n\v\f').startswith(b'"'):
        return parse_text_model(r.data, Path(path).stem)
    signature = r.string()
    if signature != 'DS2ModelFile_1':
        raise ValueError('Invalid DS2ModelFile_1 signature')
    version = r.string()
    if version == '0.6':
        return parse_binary_model(r, Path(path).stem)
    if version not in ('0.9', '1.0'):
        raise UnsupportedVersion(version)
    out = {'version': version, 'name': r.string(), 'meshes': []}
    for _ in range(r.value('B')):
        mesh = {'offset': r.pos, 'material': r.string(), 'textures': [r.string() for _ in range(5)]}
        mesh['bounds_min'] = r.array('f', 3)
        mesh['bounds_max'] = r.array('f', 3)
        mesh['primitive_count'] = r.value('I')
        mesh['triangle_strip'] = r.value('B')
        mesh['skinning_count'] = r.value('I')
        mesh['index_count'] = r.value('I')
        mesh['indices_offset'] = r.pos
        indices = r.array('H', mesh['index_count'])
        mesh['vertex_count'] = r.value('H')
        if indices and max(indices) >= mesh['vertex_count']:
            raise ValueError(f'Index outside vertex array at 0x{mesh["offset"]:X}')
        for name, stride in [('positions', 12), ('normals', 12), ('uv', 8)]:
            mesh[name + '_offset'] = r.pos
            r.take(mesh['vertex_count'] * stride)
        mesh['influence_mode'] = r.value('B')
        mesh['assignments_offset'] = r.pos
        counts = Counter()
        if mesh['influence_mode']:
            for _ in range(mesh['vertex_count']):
                count = r.value('B')
                counts[count] += 1
                r.take(count * 6)
        mesh['influence_counts'] = dict(counts)
        mesh['end'] = r.pos
        out['meshes'].append(mesh)
    out['collision_count_offset'] = r.pos
    out['collision_objects'] = []
    for _ in range(r.value('B')):
        obj = {'offset': r.pos, 'name': r.string(), 'type': r.value('B'), 'bone_index': r.value('H'), 'matrix': r.array('f', 16)}
        count = {0: 1, 1: 3, 2: 2}.get(obj['type'], 0)
        obj['parameters'] = r.array('f', count)
        obj['end'] = r.pos
        out['collision_objects'].append(obj)
    out['has_skeleton_offset'] = r.pos
    out['has_skeleton'] = r.value('B')
    if out['has_skeleton']:
        skeleton = {'offset': r.pos, 'bones': []}
        for _ in range(r.value('H')):
            bone = {'offset': r.pos, 'name': r.string(), 'local_rest_matrix': r.array('f', 16), 'parent': r.value('H'), 'children': r.indices(), 'animation_slot': r.value('H')}
            skeleton['bones'].append(bone)
        skeleton['root_bones'] = r.indices()
        skeleton['animations'] = []
        for _ in range(r.value('H')):
            anim = {'offset': r.pos, 'name': r.string(), 'duration_seconds': r.value('f'), 'time_parameter_seconds': r.value('f'), 'channels': []}
            for _ in range(r.value('H')):
                channel = {'offset': r.pos, 'bone_index': r.value('H'), 'bone_name': r.string(), 'key_count': r.value('I'), 'keys_offset': r.pos}
                # 0x100CF790: packed u8 flags; f32 quat XYZW[4], translation[3], scale[3], seconds.
                keys_start = r.take(channel['key_count'] * 45)
                flags = Counter(r.data[keys_start + i * 45] for i in range(channel['key_count']))
                channel['key_flags'] = dict(flags)
                anim['channels'].append(channel)
            anim['events'] = []
            if version == '1.0':
                for _ in range(r.value('H')):
                    anim['events'].append({'time_seconds': r.value('f'), 'name': r.string()})
            skeleton['animations'].append(anim)
        skeleton['groups'] = []
        for _ in range(r.value('H')):
            skeleton['groups'].append({'name': r.string(), 'bone_indices': r.indices(), 'animation_slots': r.indices()})
        skeleton['animation_slot_count'] = r.value('H')
        out['skeleton'] = skeleton
    out['end'] = r.pos
    out['file_size'] = len(r.data)
    if r.pos != len(r.data):
        raise ValueError(f'{len(r.data)-r.pos} trailing bytes at 0x{r.pos:X}')
    return out


def _metadata_report(model):
    """Do not expand legacy geometry or sampled palettes into CLI JSON."""
    result = dict(model)
    result['meshes'] = []
    decoded = {'positions', 'normals', 'uv_coords', 'indices',
               'matrix_tracks', 'vertex_influences'}
    for source in model['meshes']:
        mesh = {key: value for key, value in source.items() if key not in decoded}
        if 'matrix_tracks' in source:
            mesh['matrix_track_count'] = len(source['matrix_tracks'])
            mesh['influence_counts'] = dict(Counter(
                len(influences) for influences in source['vertex_influences']))
        result['meshes'].append(mesh)
    return result


def main():
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path)
    args = parser.parse_args()
    if not args.path.is_dir():
        try:
            result = parse_model(args.path)
        except (ValueError, OSError) as error:
            parser.exit(1, f'{args.path}: {error}\n')
        print(json.dumps(_metadata_report(result), ensure_ascii=False, indent=2))
        return 0
    report = {'parsed_versions': Counter(), 'unsupported_versions': Counter(),
              'mesh_count': 0, 'animation_count': 0, 'event_count': 0, 'failures': []}
    for path in sorted(args.path.rglob('*.ds2md')):
        try:
            model = parse_model(path)
        except UnsupportedVersion as error:
            report['unsupported_versions'][str(error)] += 1
            continue
        except (ValueError, OSError) as error:
            report['failures'].append({'path': str(path), 'error': str(error)})
            continue
        report['parsed_versions'][model['version']] += 1
        report['mesh_count'] += len(model['meshes'])
        animations = model.get('skeleton', {}).get('animations', [])
        report['animation_count'] += len(animations)
        report['animation_count'] += int(any(
            mesh.get('matrix_tracks') for mesh in model['meshes']))
        report['event_count'] += sum(len(a['events']) for a in animations)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return int(bool(report['failures']))


if __name__ == '__main__':
    raise SystemExit(main())
