"""DS2MD skeletal playback without a rendering dependency.

CPU matrices are row-major and act on column vectors: global = parent * local,
skin = animated_global * inverse_bind. Disk matrices are transposed on load;
Animator.matrices contains the same skin transforms in OpenGL column-major f32.
Keys are decoded only for the selected clip, not for the complete model corpus.
"""

from array import array
from bisect import bisect_left
from dataclasses import dataclass, field
import math
import struct


_IDENTITY = (1.0, 0.0, 0.0, 0.0,
             0.0, 1.0, 0.0, 0.0,
             0.0, 0.0, 1.0, 0.0,
             0.0, 0.0, 0.0, 1.0)
_KEY = struct.Struct('<B11f')
_F32_MAX = 3.4028234663852886e38


@dataclass(frozen=True, slots=True)
class Bone:
    name: str
    parent: int
    local_rest_matrix: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class Channel:
    bone_index: int
    keys_offset: int
    key_count: int


@dataclass(frozen=True, slots=True)
class Event:
    time_seconds: float
    name: str


@dataclass(frozen=True, slots=True)
class Clip:
    name: str
    duration_seconds: float
    time_parameter_seconds: float
    channels: tuple[Channel, ...]
    events: tuple[Event, ...]


@dataclass(frozen=True, slots=True)
class Skeleton:
    bones: tuple[Bone, ...]
    clips: tuple[Clip, ...]
    _data: bytes = field(repr=False)
    _order: tuple[int, ...] = field(repr=False)
    _inverse_bind: tuple[tuple[float, ...], ...] = field(repr=False)


def _finite(value, label):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f'{label} must be finite')
    return value


def _multiply(a, b):
    return tuple(a[r] * b[c] + a[r + 1] * b[c + 4]
                 + a[r + 2] * b[c + 8] + a[r + 3] * b[c + 12]
                 for r in (0, 4, 8, 12) for c in range(4))


def _finite_matrix(matrix, label):
    if not all(math.isfinite(v) for v in matrix):
        raise ValueError(f'{label} contains non-finite matrix values')
    return matrix


def _inverse(matrix, label):
    """Invert a general 4x4 bind matrix once, with partial pivoting."""
    rows = [list(matrix[r * 4:r * 4 + 4]) + list(_IDENTITY[r * 4:r * 4 + 4])
            for r in range(4)]
    for col in range(4):
        pivot = max(range(col, 4), key=lambda r: abs(rows[r][col]))
        if rows[pivot][col] == 0.0:
            raise ValueError(f'{label} has a singular bind matrix')
        rows[col], rows[pivot] = rows[pivot], rows[col]
        divisor = rows[col][col]
        rows[col] = [value / divisor for value in rows[col]]
        for r in range(4):
            if r != col:
                factor = rows[r][col]
                rows[r] = [value - factor * base
                           for value, base in zip(rows[r], rows[col])]
    return _finite_matrix(tuple(value for row in rows for value in row[4:]), label)


def load_skeleton(metadata: dict, data: bytes) -> Skeleton | None:
    """Build metadata from inspect_ds2md.parse_model, retaining packed key bytes.

    Parent order on disk is arbitrary. Validate and order it iteratively, then
    accumulate bind transforms and compute their inverses exactly once.
    """
    source = metadata.get('skeleton')
    if source is None:
        return None
    bones = []
    for index, record in enumerate(source['bones']):
        values = tuple(_finite(v, f'Bone {index} rest matrix')
                       for v in record['local_rest_matrix'])
        if len(values) != 16:
            raise ValueError(f'Bone {index} rest matrix must have 16 values')
        parent = record['parent']
        if parent == 0xFFFF:
            parent = -1
        if not isinstance(parent, int) or parent < -1 or parent >= len(source['bones']):
            raise ValueError(f'Bone {index} has invalid parent {parent!r}')
        bones.append(Bone(record['name'], parent,
                          tuple(values[c * 4 + r] for r in range(4) for c in range(4))))
    children = [[] for _ in bones]
    pending = []
    for index, bone in enumerate(bones):
        if bone.parent == -1:
            pending.append(index)
        else:
            children[bone.parent].append(index)
    order = []
    while pending:
        index = pending.pop()
        order.append(index)
        pending.extend(children[index])
    if len(order) != len(bones):
        raise ValueError('Skeleton parent hierarchy contains a cycle')
    bind = [_IDENTITY] * len(bones)
    inverse_bind = [_IDENTITY] * len(bones)
    for index in order:
        bone = bones[index]
        bind[index] = _finite_matrix(
            bone.local_rest_matrix if bone.parent == -1 else
            _multiply(bind[bone.parent], bone.local_rest_matrix), f'Bone {index}')
        inverse_bind[index] = _inverse(bind[index], f'Bone {index}')
    clips = []
    for clip_index, record in enumerate(source.get('animations', ())):
        label = f'Clip {clip_index} ({record["name"]})'
        duration = _finite(record['duration_seconds'], f'{label} duration')
        if duration < 0.0:
            raise ValueError(f'{label} duration must be nonnegative')
        parameter = _finite(record['time_parameter_seconds'], f'{label} time parameter')
        channels = []
        seen = set()
        for channel in record.get('channels', ()):
            index = channel['bone_index']
            if not isinstance(index, int) or not 0 <= index < len(bones):
                raise ValueError(f'{label} has invalid channel bone {index!r}')
            if index in seen:
                raise ValueError(f'{label} has duplicate channels for bone {index}')
            seen.add(index)
            offset, count = channel['keys_offset'], channel['key_count']
            if (not isinstance(offset, int) or not isinstance(count, int)
                    or offset < 0 or count < 0 or offset + count * _KEY.size > len(data)):
                raise ValueError(f'{label}, bone {index}: keys outside model data')
            channels.append(Channel(index, offset, count))
        events = tuple(Event(_finite(event['time_seconds'], f'{label} event time'),
                             event['name']) for event in record.get('events', ()))
        clips.append(Clip(record['name'], duration, parameter, tuple(channels), events))
    return Skeleton(tuple(bones), tuple(clips), data, tuple(order), tuple(inverse_bind))


@dataclass(slots=True)
class _DecodedChannel:
    bone_index: int
    flags: bytes
    times: array
    values: array


def _decode(skeleton, clip):
    channels = []
    for channel in clip.channels:
        if channel.key_count == 0:
            continue
        flags = bytearray()
        times = array('d')
        values = array('d')
        for index in range(channel.key_count):
            key = _KEY.unpack_from(skeleton._data, channel.keys_offset + index * _KEY.size)
            if not all(math.isfinite(v) for v in key[1:]):
                raise ValueError(f'Clip {clip.name}, bone {channel.bone_index}, '
                                 f'key {index}: non-finite value')
            time = key[11]
            if times and time <= times[-1]:
                raise ValueError(f'Clip {clip.name}, bone {channel.bone_index}: '
                                 'key times must be strictly increasing')
            norm = math.hypot(*key[1:5])
            if norm == 0.0:
                raise ValueError(f'Clip {clip.name}, bone {channel.bone_index}, '
                                 f'key {index}: zero quaternion')
            flags.append(key[0])
            times.append(time)
            values.extend(v / norm for v in key[1:5])
            values.extend(key[5:11])
        channels.append(_DecodedChannel(channel.bone_index, bytes(flags), times, values))
    return tuple(channels)


def _slerp(a, b, fraction):
    dot = sum(x * y for x, y in zip(a, b))
    if dot < 0.0:
        b = tuple(-value for value in b)
        dot = -dot
    dot = min(1.0, dot)
    if dot > 0.9995:
        q = tuple(x + (y - x) * fraction for x, y in zip(a, b))
    else:
        angle = math.acos(dot)
        divisor = math.sin(angle)
        left = math.sin((1.0 - fraction) * angle) / divisor
        right = math.sin(fraction * angle) / divisor
        q = tuple(x * left + y * right for x, y in zip(a, b))
    norm = math.hypot(*q)
    return tuple(value / norm for value in q)


def _trs(q, translation, scale):
    """Compose T*S*R: engine scale multiplies rotation rows, not columns."""
    x, y, z, w = q
    sx, sy, sz = scale
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return ((1.0 - 2.0 * (yy + zz)) * sx, 2.0 * (xy - wz) * sx,
            2.0 * (xz + wy) * sx, translation[0],
            2.0 * (xy + wz) * sy, (1.0 - 2.0 * (xx + zz)) * sy,
            2.0 * (yz - wx) * sy, translation[1],
            2.0 * (xz - wy) * sz, 2.0 * (yz + wx) * sz,
            (1.0 - 2.0 * (xx + yy)) * sz, translation[2],
            0.0, 0.0, 0.0, 1.0)


def _sample(channel, time):
    """Sample by timestamp; exact keys use their own TRS and scale-enable bit.

    Inside an open interval the RIGHT key controls hold/interpolate flags. A
    disabled right scale is identity; held scale also respects the left enable
    bit. Before/after the key range, clamp to the first/last key respectively.
    """
    times, values, flags = channel.times, channel.values, channel.flags
    right = bisect_left(times, time)
    if right == len(times):
        right -= 1
    if right == 0 or time >= times[-1] or time == times[right]:
        offset = right * 10
        return _trs(values[offset:offset + 4], values[offset + 4:offset + 7],
                    values[offset + 7:offset + 10] if flags[right] & 8 else (1.0, 1.0, 1.0))
    left = right - 1
    a, b = left * 10, right * 10
    fraction = (time - times[left]) / (times[right] - times[left])
    mode = flags[right]
    q = (values[a:a + 4] if mode & 1 else
         _slerp(values[a:a + 4], values[b:b + 4], fraction))
    translation = (values[a + 4:a + 7] if mode & 2 else
                   tuple(values[a + i] + (values[b + i] - values[a + i]) * fraction
                         for i in (4, 5, 6)))
    if not mode & 8:
        scale = (1.0, 1.0, 1.0)
    elif mode & 4:
        scale = values[a + 7:a + 10] if flags[left] & 8 else (1.0, 1.0, 1.0)
    else:
        scale = tuple(values[a + i] + (values[b + i] - values[a + i]) * fraction
                      for i in (7, 8, 9))
    return _trs(q, translation, scale)


class Animator:
    """A bind-pose-first player; seeking never wraps, even at the exact endpoint.

    Selection starts playback at zero. update() wraps on crossing a looping
    endpoint; without looping it displays the endpoint and pauses. Signed speeds
    support reverse playback. The public f32 palette array keeps its identity.
    """

    def __init__(self, skeleton: Skeleton):
        self.skeleton = skeleton
        self.clip_index = None
        self.time_seconds = 0.0
        self.playing = False
        self.loop = True
        self.speed = 1.0
        self.matrices = array('f', _IDENTITY) * len(skeleton.bones)
        self._channels = ()

    @property
    def duration_seconds(self):
        return (0.0 if self.clip_index is None else
                self.skeleton.clips[self.clip_index].duration_seconds)

    def _evaluate(self, channels, time):
        skeleton = self.skeleton
        local = [bone.local_rest_matrix for bone in skeleton.bones]
        for channel in channels:
            local[channel.bone_index] = _sample(channel, time)
        global_pose = [_IDENTITY] * len(local)
        palette = array('f', [0.0]) * (16 * len(local))
        for index in skeleton._order:
            parent = skeleton.bones[index].parent
            global_pose[index] = _finite_matrix(
                local[index] if parent == -1 else _multiply(global_pose[parent], local[index]),
                f'Animated bone {index}')
            skin = _multiply(global_pose[index], skeleton._inverse_bind[index])
            if not all(math.isfinite(v) and abs(v) <= _F32_MAX for v in skin):
                raise ValueError(f'Animated bone {index} skin matrix is not representable as f32')
            start = index * 16
            for col in range(4):
                for row in range(4):
                    palette[start + col * 4 + row] = skin[row * 4 + col]
        return palette

    def select_clip(self, index: int | None):
        if index is None:
            self.matrices[:] = array('f', _IDENTITY) * len(self.skeleton.bones)
            self._channels = ()
            self.clip_index = None
            self.time_seconds = 0.0
            self.playing = False
            return
        if not isinstance(index, int) or not 0 <= index < len(self.skeleton.clips):
            raise ValueError(f'Invalid animation clip index {index!r}')
        channels = _decode(self.skeleton, self.skeleton.clips[index])
        palette = self._evaluate(channels, 0.0)
        self.matrices[:] = palette
        self._channels = channels
        self.clip_index = index
        self.time_seconds = 0.0
        self.playing = True

    def seek(self, seconds):
        seconds = _finite(seconds, 'Seek time')
        time = min(self.duration_seconds, max(0.0, seconds))
        if self.clip_index is not None:
            self.matrices[:] = self._evaluate(self._channels, time)
        self.time_seconds = time

    def play(self):
        if self.clip_index is not None:
            self.playing = True

    def pause(self):
        self.playing = False

    def update(self, dt) -> bool:
        if not self.playing or self.clip_index is None:
            return False
        dt = _finite(dt, 'Animation delta time')
        if dt < 0.0:
            raise ValueError('Animation delta time must be nonnegative')
        speed = _finite(self.speed, 'Animation speed')
        delta = dt * speed
        if not math.isfinite(delta):
            raise ValueError('Animation time advance must be finite')
        duration = self.duration_seconds
        if duration == 0.0:
            self.playing = False
            return False
        if delta == 0.0:
            return False
        time = self.time_seconds + delta
        if not math.isfinite(time):
            raise ValueError('Animation time must be finite')
        stop = False
        if self.loop:
            if time < 0.0 or time >= duration:
                time %= duration
        else:
            if time <= 0.0 and delta < 0.0:
                time, stop = 0.0, True
            elif time >= duration and delta > 0.0:
                time, stop = duration, True
        changed = time != self.time_seconds
        if changed:
            self.matrices[:] = self._evaluate(self._channels, time)
            self.time_seconds = time
        if stop:
            self.playing = False
        return changed
