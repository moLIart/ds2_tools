"""Consumer-visible DS2 range and material parsing regressions; no game assets."""

from pathlib import Path
import struct
from tempfile import TemporaryDirectory
import unittest

from inspect_ds2 import parse_map
from ds2_scene import read_materials


def string(value):
    value = value.encode('cp1251')
    return struct.pack('<H', len(value)) + value


def chunk(name, payload):
    return string('DS2Chunk_begin') + string(name) + payload + string('DS2Chunk_end')


def geometry(*, list_count=3, list_buffer=1, strip_buffer=1, vertex_start=0, last_index=2):
    fields = (0, 2, 3, 7, list_count, list_buffer, 0, 4, strip_buffer, 3, 0, vertex_start)
    mesh = (b'\0' + string('default_lightmap') + string('example') + string('notexture')*4
            + struct.pack('<10f', 0, 0, 0, 1, 1, 1, .5, .5, .5, 1)
            + bytes(156) + b'\0' + bytes(12) + struct.pack('<12I', *fields))
    vertices = struct.pack('<2I12f', 2, 4, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0)
    indices = struct.pack('<2I7I', 1, 7, 0, 1, 2, 0, 1, 2, last_index)
    return (string('DS2GeometryFile_1') + string('0.8')
            + chunk('meshes', struct.pack('<I', 1) + mesh)
            + chunk('buffers', struct.pack('<I', 2) + vertices + indices)
            + chunk('lightmaps', struct.pack('<I', 1) + string('map\\atlas')))


class GeometryRanges(unittest.TestCase):
    def test_strip_only_allows_absent_list_buffer(self):
        # Sentinel list buffer is legal when only the strip stream is drawn.
        parse_map('memory.ds2', geometry(list_count=0, list_buffer=0xFFFFFFFF))
        with self.assertRaisesRegex(ValueError, 'index_buffer'):
            parse_map('memory.ds2', geometry(list_count=3, list_buffer=0xFFFFFFFF))

    def test_nonempty_strip_requires_its_own_valid_buffer(self):
        with self.assertRaisesRegex(ValueError, 'strip_index_buffer'):
            parse_map('memory.ds2', geometry(strip_buffer=0xFFFFFFFF))

    def test_vertex_base_participates_in_index_bounds(self):
        # Shared-buffer vertex3 is legal at base0, but becomes out of bounds
        # at base1. A mesh-local vertex_count check incorrectly rejects both.
        parse_map('memory.ds2', geometry(vertex_start=0, last_index=3))
        with self.assertRaisesRegex(ValueError, 'index plus vertex base'):
            parse_map('memory.ds2', geometry(vertex_start=1, last_index=3))

    def test_truncation_is_not_successful_partial_geometry(self):
        with self.assertRaises(ValueError):
            parse_map('memory.ds2', geometry()[:-1])

    def test_unknown_chunk_cannot_be_silently_skipped(self):
        with self.assertRaisesRegex(ValueError, 'Unknown chunk'):
            parse_map('memory.ds2', geometry() + chunk('unrecognized', bytes(8)))


class MaterialSemantics(unittest.TestCase):
    def parse(self, text):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / 'test.mat').write_text(text, encoding='cp1251')
            return read_materials(directory)

    def test_compiler_transparency_does_not_enable_blending(self):
        material = self.parse('''template_material cutout {
            ds2compiler { transparent alphashadow 0.8 }
            material lightmapped_base
            alphatest { greater 0.37 }
            diffuse_texture { $diffuse$ clampmap }
        }''')['cutout']
        self.assertEqual(material.blend, '')
        self.assertEqual(material.alpha_cutoff, .37)
        self.assertTrue(material.depth_write)
        self.assertTrue(material.clamp)

    def test_explicit_blend_and_depth_override_win_over_opaque_type(self):
        material = self.parse('''template_material glow {
            material_type opaque
            material vertlight_base
            blend add
            depth_write false
            overbright false
            color3 { 0.5 0.25 1 }
        }''')['glow']
        self.assertEqual(material.blend, 'add')
        self.assertFalse(material.depth_write)
        self.assertFalse(material.overbright)
        self.assertEqual(material.color, (.5, .25, 1.))

    def test_comment_to_eof_cannot_override_live_material(self):
        materials = self.parse('''template_material live { self_illumination true }
            /* template_material live { self_illumination false }
        ''')
        self.assertTrue(materials['live'].self_illumination)

    def test_missing_diffuse_is_untextured_not_mesh_slot_override(self):
        material = self.parse('template_material water { material space_distortion }')['water']
        self.assertEqual(material.diffuse, '$white$')


if __name__ == '__main__':
    unittest.main()
