"""Map camera movement and visible depth regressions; no window/game assets."""

import math
from types import SimpleNamespace
import unittest

from view_ds2 import MapViewer, _camera_forward, _map_clip_planes
from pyglet.math import Mat4, Vec3, Vec4
from pyglet.window import key


class MapCameraTests(unittest.TestCase):
    def test_a_and_d_strafe_in_screen_space_without_turning(self):
        for yaw in (-math.pi, -math.pi/2, 0., math.pi/2, .9*math.pi):
            for pitch in (-1., 0., 1.):
                with self.subTest(yaw=yaw, pitch=pitch):
                    forward = _camera_forward(yaw, pitch)
                    view = Mat4.look_at(Vec3(), forward, Vec3(0, 0, 1))
                    keys = key.KeyStateHandler()
                    state = SimpleNamespace(fly=True, scene=True, eye=Vec3(),
                                            yaw=yaw, pitch=pitch, speed=100.,
                                            keys=keys, invalid=False)
                    keys.on_key_press(key.A, 0)
                    MapViewer.update(state, .05)
                    left = state.eye
                    self.assertLess((view @ Vec4(*left, 0)).x, 0.)
                    self.assertAlmostEqual(left.dot(forward), 0.)
                    keys.on_key_release(key.A, 0)
                    keys.on_key_press(key.D, 0)
                    MapViewer.update(state, .05)
                    right = state.eye-left
                    self.assertGreater((view @ Vec4(*right, 0)).x, 0.)
                    self.assertAlmostEqual(state.eye.length(), 0.)
                    self.assertEqual((state.yaw, state.pitch), (yaw, pitch))

    def assert_visible_depth(self, eye, forward, low, high, depth):
        near, far = _map_clip_planes(eye, forward, low, high)
        projection = Mat4.perspective_projection(1., near, far, fov=50)
        view = Mat4.look_at(eye, eye+forward, Vec3(0, 0, 1))
        clip = projection @ view @ Vec4(*(eye+forward*depth), 1)
        self.assertGreater(clip.z/clip.w, -1.)
        self.assertLess(clip.z/clip.w, 1.)

    def test_zoom_inside_large_map_retains_nearby_surfaces(self):
        low, high = Vec3(-100000, -100000, -100000), Vec3(100000, 100000, 100000)
        for distance in (20000., 10000., 1000., 1.):
            with self.subTest(distance=distance):
                self.assert_visible_depth(Vec3(0, -distance, 0), Vec3(0, 1, 0),
                                          low, high, 1.)

    def test_flight_past_orbit_target_keeps_geometry_ahead(self):
        eye = Vec3(20, 0, 0)
        low, high = Vec3(-100, -100, -100), Vec3(100, 100, 100)
        # Looking farther along +X after crossing the old target at the origin.
        self.assert_visible_depth(eye, Vec3(1, 0, 0), low, high, 79.)
        # Rotation in place must refit far using the new viewing direction.
        self.assert_visible_depth(eye, Vec3(0, 1, 0), low, high, 99.)
        self.assert_visible_depth(eye, Vec3(-1, 0, 0), low, high, 119.)

    def test_flying_away_expands_far_for_the_visible_map(self):
        low, high = Vec3(-100, -100, -100), Vec3(100, 100, 100)
        for y in (-1000., -10000., -100000.):
            with self.subTest(y=y):
                self.assert_visible_depth(Vec3(0, y, 0), Vec3(0, 1, 0),
                                          low, high, 99.-y)

    def test_outside_bounds_keeps_tight_depth_precision(self):
        near, far = _map_clip_planes(Vec3(0, -10000, 0), Vec3(0, 1, 0),
                                     Vec3(-100, -100, -100), Vec3(100, 100, 100))
        self.assertLess(near, 9900.)
        self.assertGreater(far, 10100.)
        self.assertLess(far/near, 1.1)

    def test_draw_distance_caps_depth_even_when_map_is_beyond_it(self):
        eye, forward = Vec3(0, -10000, 0), Vec3(0, 1, 0)
        low, high = Vec3(-100, -100, -100), Vec3(100, 100, 100)
        view = Mat4.look_at(eye, eye+forward, Vec3(0, 0, 1))
        for limit in (1., 100., 10000.):
            with self.subTest(limit=limit):
                near, far = _map_clip_planes(eye, forward, low, high, draw_distance=limit)
                self.assertEqual(far, limit)
                self.assertGreater(near, 0.)
                self.assertLess(near, far)
                projection = Mat4.perspective_projection(1., near, far, fov=50)
                inside = projection @ view @ Vec4(*(eye+forward*((near+far)*.5)), 1)
                outside = projection @ view @ Vec4(*(eye+forward*(far+1.)), 1)
                self.assertGreater(inside.z/inside.w, -1.)
                self.assertLess(inside.z/inside.w, 1.)
                self.assertGreater(outside.z/outside.w, 1.)


if __name__ == '__main__':
    unittest.main()
