"""CPU occlusion grouping regressions; no window, Pyglet, or game assets."""

import math
import unittest

from ds2_occlusion import OcclusionGroup, build_occlusion_groups, group_query_safe


def mesh(low, high, **extra):
    return dict(bounds_min=low, bounds_max=high, **extra)


class OcclusionGroupTests(unittest.TestCase):
    def assert_partition(self, meshes, groups, group_size):
        ownership = [index for group in groups for index in group.mesh_indices]
        self.assertEqual(sorted(ownership), list(range(len(meshes))))
        for group in groups:
            self.assertLessEqual(len(group.mesh_indices), group_size)
            for index in group.mesh_indices:
                for axis in range(3):
                    self.assertLess(group.bounds_min[axis], meshes[index]['bounds_min'][axis])
                    self.assertGreater(group.bounds_max[axis], meshes[index]['bounds_max'][axis])

    def test_unique_ownership_and_complete_overlapping_world_bounds(self):
        meshes = [mesh((-10000, -20000, -30000), (10000, 20000, 30000))]
        meshes += [mesh((i-100, -i, 2), (i+100, i+1, 2)) for i in range(1, 25)]
        groups = build_occlusion_groups(meshes, group_size=3)
        self.assert_partition(meshes, groups, 3)

    def test_serialized_instance_bounds_are_not_transformed_again(self):
        meshes = [mesh((1000, -2000, 3000), (1010, -1990, 3010),
                       matrix=(100., 0., 0., 0., 0., 100., 0., 0.,
                               0., 0., 100., 0., 9000., 9000., 9000., 1.)),
                  mesh((-5000, 4000, 9), (-5000, 4000, 9))]
        groups = build_occlusion_groups(meshes, group_size=1)
        self.assert_partition(meshes, groups, 1)

    def test_partition_is_deterministic_including_identical_centroids(self):
        meshes = [mesh((-i, -i*2, -i*3), (i, i*2, i*3)) for i in range(80)]
        self.assertEqual(build_occlusion_groups(meshes, 7), build_occlusion_groups(meshes, 7))
        self.assert_partition(meshes, build_occlusion_groups(meshes, 7), 7)
        self.assertEqual(build_occlusion_groups([]), ())
        with self.assertRaises(ValueError):
            build_occlusion_groups(meshes, 0)

    def test_camera_inside_and_box_boundary_are_not_query_safe(self):
        group = OcclusionGroup((0,), (-1., -1., -1.), (1., 1., 1.))
        for eye in ((0., 0., 0.), (1., 0., 0.), (-1., -1., -1.)):
            with self.subTest(eye=eye):
                self.assertFalse(group_query_safe(group, eye, (1., 0., 0.), .1))

    def test_entire_box_must_clear_near_plane_with_tolerance(self):
        for low, high in (((.5, -1, -1), (2, 1, 1)),
                          ((1, -1, -1), (2, 1, 1)),
                          ((1+1e-8, -1, -1), (2, 1, 1)),
                          ((-3, -1, -1), (-2, 1, 1))):
            with self.subTest(low=low):
                group = OcclusionGroup((0,), low, high)
                self.assertFalse(group_query_safe(group, (0, 0, 0), (1, 0, 0), 1.))
        group = OcclusionGroup((0,), (2, -1, -1), (3, 1, 1))
        self.assertTrue(group_query_safe(group, (0, 0, 0), (1, 0, 0), 1.))
        self.assertFalse(group_query_safe(group, (0, 0, 0), (-1, 0, 0), 1.))

    def test_rotated_forward_uses_nearest_corner_not_box_center(self):
        forward = (1/math.sqrt(2), -1/math.sqrt(2), 0.)
        group = OcclusionGroup((0,), (3., 0., -1.), (5., 2., 1.))
        # The center clears near=1, but the nearest corner has depth 1/sqrt(2).
        self.assertFalse(group_query_safe(group, (0, 0, 0), forward, 1.))
        self.assertTrue(group_query_safe(group, (0, 0, 0), forward, .5))
        self.assertFalse(group_query_safe(group, (0, 0, 0), forward, 1/math.sqrt(2)))
        self.assertTrue(group_query_safe(group, (-10, 10, 0), forward, 1.))


if __name__ == '__main__':
    unittest.main()
