"""Clip-plane math regressions; no window or game resources are created."""

import math
import unittest

from view_ds2md import _clip_planes


class CameraDepthTests(unittest.TestCase):
    def test_outside_bounds_keeps_surfaces_and_uses_available_precision(self):
        near, far = _clip_planes((0., -10., 0.), (0., 0., 0.), 10.,
                                 (-1., -1., -1.), (1., 1., 1.))
        self.assertLess(near, 9.)
        self.assertGreater(far, 11.)
        self.assertLess(far / near, 2.)

    def test_entering_animated_bounds_does_not_collapse_near(self):
        planes = [_clip_planes((0., -0.5, 0.), (0., 0., 0.), 0.5,
                               (-extent, -extent, -extent), (extent, extent, extent))
                  for extent in (0.49, 0.5, 0.51, 1.)]
        for near, far in planes:
            self.assertGreater(near, 0.)
            self.assertGreater(far, near)
            self.assertLess(far / near, 1000.)
        self.assertLess(max(near for near, _ in planes) /
                        min(near for near, _ in planes), 3.)

    def test_crossing_old_bounding_sphere_boundary_has_no_near_jump(self):
        radius = math.sqrt(3.)
        planes = [_clip_planes((0., -distance, 0.), (0., 0., 0.), distance,
                               (-1., -1., -1.), (1., 1., 1.))
                  for distance in (radius - 1e-5, radius, radius + 1e-5)]
        self.assertLess(max(near for near, _ in planes) /
                        min(near for near, _ in planes), 1.001)

    def test_oblique_view_contains_every_positive_bound_corner(self):
        eye = (5., -7., 4.)
        distance = math.sqrt(sum(value * value for value in eye))
        near, far = _clip_planes(eye, (0., 0., 0.), distance,
                                 (-2., -3., -1.), (2., 3., 1.))
        for x in (-2., 2.):
            for y in (-3., 3.):
                for z in (-1., 1.):
                    depth = sum((coordinate - origin) * -origin / distance
                                for coordinate, origin in zip((x, y, z), eye))
                    self.assertLess(near, depth)
                    self.assertGreater(far, depth)

    def test_geometry_behind_eye_still_produces_valid_projection(self):
        near, far = _clip_planes((0., 0., 0.), (0., 1., 0.), 1.,
                                 (-1., -10., -1.), (1., -5., 1.))
        self.assertGreater(near, 0.)
        self.assertGreater(far, near)
        self.assertTrue(math.isfinite(near) and math.isfinite(far))

    def test_depth_precision_is_independent_of_model_units(self):
        reference = _clip_planes((0., -0.5, 0.), (0., 0., 0.), 0.5,
                                  (-1., -1., -1.), (1., 1., 1.))
        for scale in (1e-6, 1., 1e6):
            with self.subTest(scale=scale):
                near, far = _clip_planes((0., -0.5 * scale, 0.), (0., 0., 0.),
                                         0.5 * scale, (-scale, -scale, -scale),
                                         (scale, scale, scale))
                self.assertAlmostEqual(near / scale, reference[0])
                self.assertAlmostEqual(far / scale, reference[1])


if __name__ == '__main__':
    unittest.main()
