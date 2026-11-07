"""Isolated playback regressions; no game files or GL context required."""

import math
import struct
import unittest

from ds2_research.ds2md_animation import Animator, load_skeleton


IDENTITY = (1., 0., 0., 0., 0., 1., 0., 0.,
            0., 0., 1., 0., 0., 0., 0., 1.)


def bone(parent=0xFFFF, translation=(0., 0., 0.)):
    matrix = list(IDENTITY)
    matrix[12:15] = translation  # Serialized order, before engine transpose.
    return {'name': 'bone', 'parent': parent, 'local_rest_matrix': matrix}


def key(time, *, flags=0, rotation=(0., 0., 0., 1.),
        translation=(0., 0., 0.), scale=(1., 1., 1.)):
    return struct.pack('<B11f', flags, *rotation, *translation, *scale, time)


def player(bones, tracks, duration=1., events=()):
    data = bytearray()
    channels = []
    for index, keys in tracks.items():
        channels.append({'bone_index': index, 'keys_offset': len(data),
                         'key_count': len(keys)})
        data.extend(b''.join(keys))
    metadata = {'skeleton': {'bones': bones, 'animations': [
        {'name': 'clip', 'duration_seconds': duration, 'time_parameter_seconds': 37.,
         'channels': channels, 'events': events}]}}
    return Animator(load_skeleton(metadata, bytes(data)))


def point(animator, value=(0., 0., 0.), index=0):
    matrix = animator.matrices[index * 16:(index + 1) * 16]
    return tuple(sum(matrix[col * 4 + row] * value[col] for col in range(3))
                 + matrix[12 + row] for row in range(3))


class AnimationPlaybackTests(unittest.TestCase):
    def assertPoint(self, actual, expected):
        for a, b in zip(actual, expected):
            self.assertAlmostEqual(a, b, places=5)

    def test_hierarchy_and_inverse_bind_when_parent_follows_child(self):
        turn = (0., 0., math.sqrt(0.5), math.sqrt(0.5))
        animator = player([bone(1, (0., 2., 0.)), bone(translation=(10., 0., 0.))],
                          {0: [key(0., translation=(0., 2., 0.))],
                           1: [key(0., rotation=turn, translation=(20., 0., 0.))]})
        self.assertPoint(point(animator, (10., 2., 0.)), (10., 2., 0.))
        animator.select_clip(0)
        self.assertPoint(point(animator, (10., 2., 0.)), (18., 0., 0.))
        self.assertPoint(point(animator, (10., 0., 0.), 1), (20., 0., 0.))
        animator.select_clip(None)
        self.assertPoint(point(animator, (10., 2., 0.)), (10., 2., 0.))
        self.assertFalse(animator.playing)

    def test_missing_channel_keeps_local_rest_under_animated_parent(self):
        animator = player([bone(), bone(0, (0., 2., 0.))],
                          {0: [key(0., translation=(5., 0., 0.))]})
        animator.select_clip(0)
        self.assertPoint(point(animator, (0., 2., 0.), 1), (5., 2., 0.))

    def test_nonuniform_scale_multiplies_rotation_rows(self):
        animator = player([bone()], {0: [key(0., flags=8,
                          rotation=(0., 0., math.sqrt(0.5), math.sqrt(0.5)),
                          scale=(2., 3., 4.), translation=(5., 6., 7.))]})
        animator.select_clip(0)
        self.assertPoint(point(animator, (1., 0., 0.)), (5., 9., 7.))
        self.assertPoint(point(animator, (0., 1., 0.)), (3., 6., 7.))

    def test_right_key_holds_components_until_exact_endpoint(self):
        animator = player([bone()], {0: [key(0., flags=8, scale=(2., 2., 2.)),
                          key(1., flags=15, rotation=(0., 0., 1., 0.),
                              translation=(10., 0., 0.), scale=(4., 4., 4.))]})
        animator.select_clip(0)
        animator.seek(0.5)
        self.assertPoint(point(animator, (1., 0., 0.)), (2., 0., 0.))
        animator.seek(1.)
        self.assertPoint(point(animator, (1., 0., 0.)), (6., 0., 0.))

    def test_left_hold_flags_do_not_control_next_interval(self):
        animator = player([bone()], {0: [key(0., flags=3),
                          key(1., rotation=(0., 0., 1., 0.),
                              translation=(10., 0., 0.))]})
        animator.select_clip(0)
        animator.seek(0.5)
        self.assertPoint(point(animator, (1., 0., 0.)), (5., 1., 0.))

    def test_disabled_scale_and_held_left_scale_enable(self):
        for right_flags in (0, 12):
            with self.subTest(right_flags=right_flags):
                animator = player([bone()], {0: [key(0., scale=(5., 5., 5.)),
                                  key(1., flags=right_flags, scale=(9., 9., 9.))]})
                animator.select_clip(0)
                animator.seek(0.5)
                self.assertPoint(point(animator, (1., 0., 0.)), (1., 0., 0.))
                animator.seek(1.)
                self.assertPoint(point(animator, (1., 0., 0.)),
                                 (9., 0., 0.) if right_flags & 8 else (1., 0., 0.))

    def test_key_timestamps_control_interpolation_and_clamping(self):
        animator = player([bone()], {0: [key(0.2, translation=(2., 0., 0.)),
                          key(0.8, translation=(8., 0., 0.))]})
        animator.select_clip(0)
        self.assertPoint(point(animator), (2., 0., 0.))
        animator.seek(0.5)
        self.assertPoint(point(animator), (5., 0., 0.))
        animator.seek(1.)
        self.assertPoint(point(animator), (8., 0., 0.))

    def test_quaternion_normalization_and_shortest_path(self):
        animator = player([bone()], {0: [key(0., rotation=(0., 0., 0., 2.)),
                          key(1., rotation=(0., 0., 0., -4.))]})
        animator.select_clip(0)
        animator.seek(0.5)
        self.assertPoint(point(animator, (1., 2., 3.)), (1., 2., 3.))

    def test_pause_speed_loop_seek_and_nonloop_endpoint(self):
        animator = player([bone()], {0: [key(0.), key(1., translation=(10., 0., 0.))]})
        animator.select_clip(0)
        animator.speed = 2.
        animator.update(0.2)
        self.assertPoint(point(animator), (4., 0., 0.))
        animator.pause()
        self.assertFalse(animator.update(10.))
        self.assertAlmostEqual(animator.time_seconds, 0.4)
        animator.seek(1.)  # Scrubbing to the endpoint must not wrap.
        self.assertPoint(point(animator), (10., 0., 0.))
        animator.play()
        animator.update(0.125)
        self.assertPoint(point(animator), (2.5, 0., 0.))
        animator.loop = False
        animator.update(0.5)
        self.assertPoint(point(animator), (10., 0., 0.))
        self.assertFalse(animator.playing)

    def test_invalid_clip_selection_does_not_replace_active_pose(self):
        animator = player([bone()], {0: [key(0., translation=(3., 0., 0.))]})
        animator.select_clip(0)
        with self.assertRaises(ValueError):
            animator.select_clip(2)
        self.assertPoint(point(animator), (3., 0., 0.))
        self.assertEqual(animator.clip_index, 0)

    def test_cyclic_or_singular_bind_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'cycle'):
            player([bone(1), bone(0)], {})
        singular = bone()
        singular['local_rest_matrix'][0] = 0.
        with self.assertRaisesRegex(ValueError, 'singular'):
            player([singular], {})

    def test_invalid_key_is_rejected_when_selected(self):
        animator = player([bone()], {0: [key(0., rotation=(0., 0., 0., 0.))]})
        with self.assertRaisesRegex(ValueError, 'quaternion'):
            animator.select_clip(0)
        self.assertIsNone(animator.clip_index)
        self.assertPoint(point(animator, (1., 2., 3.)), (1., 2., 3.))


if __name__ == '__main__':
    unittest.main()
