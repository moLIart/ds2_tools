"""Native DS2 0.8 map viewer: albedo, baked lightmaps and vertex lighting.

Run: python ds2_research/view_ds2.py [MAP.ds2]
Requires the same requirements-viewer.txt and OpenGL 3.3 as the model viewer.
"""

import argparse
import ctypes
from dataclasses import dataclass
import math
from pathlib import Path
import sys

import pyglet
pyglet.options['shadow_window'] = False
from pyglet import gl
from pyglet.graphics.shader import Shader, ShaderProgram
from pyglet.math import Mat4, Vec3
from pyglet.window import key, mouse

from ds2_scene import TextureIndex, load_map_scene, read_materials, visible_mesh_indices
from view_ds2md import choose_file, load_albedo_texture


VERTEX = """#version 330 core
layout(location=0) in vec3 position;
layout(location=1) in vec2 uv;
layout(location=2) in vec2 lm_uv;
layout(location=3) in uvec4 packed_color;
uniform mat4 mvp;
uniform mat4 model;
out vec2 albedo_uv;
out vec2 lightmap_uv;
out vec4 vertex_color;
void main() {
    gl_Position = mvp * model * vec4(position, 1.0);
    albedo_uv = uv;
    lightmap_uv = lm_uv;
    vertex_color = vec4(packed_color & uvec4(255u)) / 255.0;
}
"""
FRAGMENT = """#version 330 core
in vec2 albedo_uv;
in vec2 lightmap_uv;
in vec4 vertex_color;
uniform sampler2D albedo;
uniform sampler2D lightmap;
uniform bool flip_albedo;
uniform bool flip_lightmap;
uniform bool use_lightmap;
uniform bool use_colors;
uniform bool has_colors;
uniform vec3 tint;
uniform vec3 mesh_color;
uniform float alpha_cutoff;
uniform float gain;
uniform int mode;
out vec4 final_color;
void main() {
    if (mode == 4) {
        final_color = vec4(mesh_color, 1.0);
        return;
    }
    vec2 uv = vec2(albedo_uv.x, flip_albedo ? 1.0-albedo_uv.y : albedo_uv.y);
    vec2 luv = vec2(lightmap_uv.x, flip_lightmap ? 1.0-lightmap_uv.y : lightmap_uv.y);
    vec4 base = texture(albedo, uv);
    vec3 lighting = vec3(1.0);
    if (use_lightmap) lighting *= texture(lightmap, luv).rgb;
    if (use_colors && has_colors) lighting *= vertex_color.rgb;
    float alpha = base.a * (use_colors && has_colors ? vertex_color.a : 1.0);
    if (alpha <= alpha_cutoff) discard;
    if (mode == 0) final_color = vec4(base.rgb * tint * lighting * gain, alpha);
    else if (mode == 1) final_color = vec4(base.rgb * tint, 1.0);
    else if (mode == 2) final_color = vec4(use_lightmap ? texture(lightmap, luv).rgb : vec3(0.4), 1.0);
    else final_color = vec4(has_colors ? vertex_color.rgb : vec3(0.4), 1.0);
}
"""
MODES = ('Combined', 'Albedo', 'Lightmap', 'Vertex colors', 'Mesh colors')
IDENTITY = (1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.)


def _mesh_color(mesh_index):
    """An odd multiplier permutes RGB24: no repeats for 24-bit mesh indices."""
    rgb = (mesh_index*0x9E3779 + 0x4F9CFF) & 0xFFFFFF
    return ((rgb >> 16)/255., ((rgb >> 8) & 255)/255., (rgb & 255)/255.)


def _camera_forward(yaw, pitch):
    return Vec3(-math.cos(pitch)*math.cos(yaw),
                -math.cos(pitch)*math.sin(yaw), -math.sin(pitch))


def _map_clip_planes(eye, forward, bounds_min, bounds_max, draw_distance=None):
    """Fit depth along the actual view, independent of the old orbit target.

    Outside the map, a tight near plane preserves depth precision. Inside its
    bounds, keep nearby surfaces instead of using a fraction of orbit distance.
    An optional draw distance caps forward depth, not radial mesh distance.
    """
    center_depth = sum(((low+high)*.5-eye[axis])*forward[axis]
                       for axis, (low, high) in enumerate(zip(bounds_min, bounds_max)))
    half_depth = sum((high-low)*.5*abs(forward[axis])
                     for axis, (low, high) in enumerate(zip(bounds_min, bounds_max)))
    padding = max(1., half_depth*.01)
    near = max(.5, center_depth-half_depth-padding)
    far = max(near+1., center_depth+half_depth+padding)
    if draw_distance is not None:
        far = min(far, draw_distance)
        # The map can lie entirely beyond the cap, including its fitted near.
        near = min(near, far-.5)
    return near, far


def _mesh_distance_squared(mesh, eye):
    """Distance to the world-space AABB center; instance transforms are baked in."""
    low, high = mesh['bounds_min'], mesh['bounds_max']
    x = (low[0]+high[0])*.5-eye[0]
    y = (low[1]+high[1])*.5-eye[1]
    z = (low[2]+high[2])*.5-eye[2]
    return x*x + y*y + z*z


@dataclass(frozen=True, slots=True)
class _GeometryBinding:
    vao: gl.GLuint
    has_uv: bool
    has_lightmap_uv: bool
    has_colors: bool


class MapViewer(pyglet.window.Window):
    def __init__(self, path=None, *, mode=0, screenshot=None):
        config = gl.Config(double_buffer=True, depth_size=24, major_version=3, minor_version=3)
        self._camera_buttons = 0
        super().__init__(width=1280, height=850, caption='DS2 Map Viewer',
                         resizable=True, config=config, file_drops=True, vsync=True)
        self.set_minimum_size(760, 480)
        self.scene = None
        self.buffer = gl.GLuint()
        self.geometry_bindings = {}
        self.program = ShaderProgram(Shader(VERTEX, 'vertex'), Shader(FRAGMENT, 'fragment'))
        self.locations = {name: gl.glGetUniformLocation(self.program.id, name.encode()) for name in (
            'model', 'flip_albedo', 'flip_lightmap', 'use_lightmap', 'use_colors',
            'has_colors', 'tint', 'mesh_color', 'alpha_cutoff', 'gain', 'mode')}
        self.textures = {}
        self.opaque, self.transparent = [], []
        self.gpu_buffers = []
        self.visible_mesh_count = 0
        self.load_error = False
        self.mode = mode
        self.wireframe = False
        self.fly = False
        self.speed = 200.0
        self.draw_distance = None
        self.eye = Vec3(0, 0, 0)
        self.target = Vec3(0, 0, 0)
        self.yaw, self.pitch, self.distance, self.radius = -.9, .4, 10., 1.
        self.keys = key.KeyStateHandler()
        self.push_handlers(self.keys)
        game = Path(__file__).resolve().parent.parent
        self.texture_index = TextureIndex(game / 'gameres' / 'textures')
        self.materials = read_materials(game / 'gameres' / 'materials')
        self.map_paths = sorted((game / 'gameres' / 'maps').rglob('*.ds2'))
        self.screenshot = Path(screenshot) if screenshot else None
        self.pending = None
        self.ui = pyglet.graphics.Batch()
        self.title = pyglet.text.Label('DS2 Map Viewer', x=16, font_size=14, batch=self.ui)
        self.stats = pyglet.text.Label('', x=16, font_size=10, batch=self.ui)
        self.status = pyglet.text.Label('Open or drop a .ds2 map.', x=16, y=12, font_size=9,
                                        batch=self.ui, color=(190, 200, 214, 255))
        self.controls = pyglet.text.Label(
            'LMB orbit | MMB pan | Wheel zoom/speed | RMB fly-look | F fly/orbit | WASD, Q/E move | Shift fast\n'
            '1 combined | 2 albedo | 3 lightmap | 4 vertex colors | 5 mesh colors | G wireframe | Home fit | [ / ] previous/next map | Ctrl+O open',
            x=16, y=64, font_size=9, multiline=True, width=1240, batch=self.ui)
        self.distance_label = pyglet.text.Label('', x=16, y=32, font_size=9, batch=self.ui)
        self.buttons = []
        for label, x, width, action in [('Open...', 16, 90, self.open_file),
                                       ('< Map', 114, 80, lambda: self.change_map(-1)),
                                       ('Map >', 202, 80, lambda: self.change_map(1)),
                                       ('Fit', 290, 70, self.frame_map)]:
            rect = pyglet.shapes.Rectangle(x, 0, width, 28, color=(48, 69, 90), batch=self.ui)
            text = pyglet.text.Label(label, x=x+width/2, anchor_x='center', anchor_y='center',
                                      font_size=10, batch=self.ui)
            self.buttons.append((rect, text, action))
        self.mode_label = pyglet.text.Label('', x=380, font_size=10, batch=self.ui)
        self._layout()
        self.constants = {}
        for name, rgba in [('$white$', (255, 255, 255, 255)), ('$black$', (0, 0, 0, 255)),
                           ('$missing$', (255, 0, 255, 255))]:
            image = pyglet.image.ImageData(1, 1, 'RGBA', bytes(rgba))
            self.constants[name] = image.get_texture()
        gl.glClearColor(.055, .07, .09, 1)
        pyglet.clock.schedule_interval(self.update, 1/60)
        initial = Path(path) if path else (self.map_paths[0] if self.map_paths else None)
        if initial:
            self.queue_load(initial)

    def _layout(self):
        self.title.y = self.height - 26
        self.stats.y = self.height - 50
        self.controls.width = self.width - 32
        self.status.width = self.width - 32
        for rect, text, _ in self.buttons:
            rect.y = self.height - 88
            text.y = self.height - 74
        self.mode_label.y = self.height - 80
        self.invalid = True

    def _status(self, text, error=False):
        self.status.text = text
        self.status.color = (255, 160, 140, 255) if error else (190, 200, 214, 255)
        self.invalid = True

    def queue_load(self, path):
        if self.pending is not None:
            pyglet.clock.unschedule(self._load_pending)
        self.pending = Path(path)
        self._status(f'Loading {self.pending.name}...')
        pyglet.clock.schedule_once(self._load_pending, .05)

    def _texture(self, source, loaded, warnings):
        if isinstance(source, str) and source in self.constants:
            return self.constants[source]
        if source is None:
            return self.constants['$missing$']
        if source not in loaded:
            try:
                loaded[source] = load_albedo_texture(source)
            except Exception as error:
                warnings.append(f'{source.name}: {error}')
                loaded[source] = None
        return loaded[source] or self.constants['$missing$']

    def _load_pending(self, dt):
        path, self.pending = self.pending, None
        loaded, warnings = {}, []
        new_buffer = gl.GLuint()
        new_bindings = {}
        try:
            scene = load_map_scene(path, self.texture_index, self.materials)
            warnings.extend(scene.warnings)
            gl.glGenBuffers(1, ctypes.byref(new_buffer))
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, new_buffer)
            # Disk arrays are packed at arbitrary byte offsets. Attribute and
            # index fetches need aligned GPU offsets: repack buffer payloads,
            # not vertices, preserving all shared ranges without CPU copies.
            gpu_buffers, uploads, size = [], [], 0
            for source in scene.metadata['buffers']:
                start, end = source['offset'] + 8, source['end']
                size = (size + 15) & ~15
                gpu = dict(source)
                for field in source:
                    if field.endswith('_offset'):
                        gpu[field] = size + source[field] - start
                gpu_buffers.append(gpu)
                uploads.append((size, start, end - start))
                size += end - start
            gl.glBufferData(gl.GL_ARRAY_BUFFER, size, None, gl.GL_STATIC_DRAW)
            address = ctypes.cast(ctypes.c_char_p(scene.data), ctypes.c_void_p).value
            for destination, start, length in uploads:
                gl.glBufferSubData(gl.GL_ARRAY_BUFFER, destination, length, ctypes.c_void_p(address + start))
            opaque, transparent = [], []
            for mesh_index, mesh in enumerate(scene.metadata['meshes']):
                material = scene.materials[mesh['material'].casefold()]
                reference = mesh['textures'][0] if material.diffuse == '$diffuse$' else material.diffuse
                albedo_source = scene.albedos[reference]
                albedo = self._texture(albedo_source, loaded, warnings)
                lm_index = mesh['lightmap_index']
                lm_source = scene.lightmaps[lm_index] if lm_index >= 0 else None
                lightmap = self._texture(lm_source, loaded, warnings) if lm_index >= 0 else self.constants['$white$']
                binding = self._prepare_geometry(mesh, gpu_buffers, new_buffer, new_bindings)
                matrix = (gl.GLfloat * 16)(*(mesh.get('matrix', IDENTITY)))
                entry = (mesh, material, albedo, lightmap,
                         isinstance(albedo_source, Path) and albedo_source.suffix.casefold() == '.tga',
                         lm_source is not None and lm_source.suffix.casefold() == '.tga',
                         matrix, mesh_index, binding, _mesh_color(mesh_index))
                (transparent if material.blend else opaque).append(entry)
            gl.glBindVertexArray(0)
        except Exception as error:
            gl.glBindVertexArray(0)
            for binding in new_bindings.values():
                gl.glDeleteVertexArrays(1, ctypes.byref(binding.vao))
            if new_buffer.value:
                gl.glDeleteBuffers(1, ctypes.byref(new_buffer))
            for texture in loaded.values():
                if texture is not None:
                    texture.delete()
            self._status(f'Cannot open {path.name}: {error}', True)
            self.load_error = True
            print(self.status.text, file=sys.stderr)
            if self.screenshot:
                self.close()
            return
        self._release_scene()
        self.scene, self.buffer, self.textures = scene, new_buffer, loaded
        self.gpu_buffers = gpu_buffers
        self.geometry_bindings = new_bindings
        self.load_error = False
        self.opaque, self.transparent = opaque, transparent
        self.frame_map()
        self.set_caption(f'{scene.path.name} - DS2 Map Viewer')
        self.title.text = str(scene.path)
        vertices = sum(b['count'] for b in scene.metadata['buffers'] if 'positions_offset' in b)
        primitives = sum(m['primitive_count'] for m in scene.metadata['meshes'])
        self.stats.text = (f'DS2 {scene.metadata["version"]} | {len(scene.metadata["meshes"]):,} meshes | '
                           f'{vertices:,} stored vertices | {primitives:,} primitives | '
                           f'{len(scene.albedos)} albedos | {len(scene.lightmaps)} lightmaps')
        for warning in warnings:
            print(f'{path.name}: {warning}', file=sys.stderr)
        self._status(f'{len(warnings)} warnings (console): {warnings[0]}' if warnings else 'All referenced textures loaded.', bool(warnings))
        if self.screenshot:
            pyglet.clock.schedule_once(self._capture, .25)

    def _release_scene(self):
        for binding in self.geometry_bindings.values():
            gl.glDeleteVertexArrays(1, ctypes.byref(binding.vao))
        self.geometry_bindings.clear()
        if self.buffer.value:
            gl.glDeleteBuffers(1, ctypes.byref(self.buffer))
            self.buffer = gl.GLuint()
        for texture in self.textures.values():
            if texture is not None:
                texture.delete()
        self.textures.clear()
        self.opaque.clear()
        self.transparent.clear()

    def frame_map(self):
        if not self.scene:
            return
        lo, hi = self.scene.bounds_min, self.scene.bounds_max
        self.target = Vec3(*((a+b)*.5 for a, b in zip(lo, hi)))
        self.radius = max(math.dist(lo, hi)*.5, 1.)
        aspect = self.width / max(1, self.height-180)
        angle = min(math.radians(50)/2, math.atan(math.tan(math.radians(50)/2)*aspect))
        self.distance = self.radius / math.sin(angle)*1.1
        self.yaw, self.pitch, self.fly = -.95, .45, False
        self.speed = max(10., self.radius*.1)
        self.invalid = True

    def _camera(self):
        forward = _camera_forward(self.yaw, self.pitch)
        if self.fly:
            return self.eye, Mat4.look_at(self.eye, self.eye+forward, Vec3(0, 0, 1))
        eye = self.target-forward*self.distance
        return eye, Mat4.look_at(eye, self.target, Vec3(0, 0, 1))

    def _set_fly(self, enabled):
        if enabled and not self.fly:
            self.eye = self._camera()[0]
        elif self.fly and not enabled:
            self.target = self.eye+_camera_forward(self.yaw, self.pitch)*self.distance
        self.fly = enabled
        self.invalid = True

    def _adjust_draw_distance(self, factor):
        if self.draw_distance is None:
            if not self.scene:
                return
            eye = self._camera()[0]
            _, distance = _map_clip_planes(
                eye, _camera_forward(self.yaw, self.pitch),
                self.scene.bounds_min, self.scene.bounds_max)
        else:
            distance = self.draw_distance
        self.draw_distance = max(1., min(1e8, distance*factor))
        self.invalid = True

    def _prepare_geometry(self, mesh, buffers, buffer, bindings):
        vertex_buffer, base = mesh['vertex_buffer'], mesh['vertex_start']
        color_buffer, color_base = vertex_buffer, base
        if mesh['instanced'] and mesh['instance_fields'][0] != 0xFFFFFFFF:
            color_buffer, color_base = mesh['instance_fields']
        key = vertex_buffer, base, color_buffer, color_base
        binding = bindings.get(key)
        if binding is not None:
            return binding
        vb, colors = buffers[vertex_buffer], buffers[color_buffer]
        binding = _GeometryBinding(gl.GLuint(), 'uv_offset' in vb,
                                   'lightmap_uv_offset' in vb, 'colors_offset' in colors)
        bindings[key] = binding
        gl.glGenVertexArrays(1, ctypes.byref(binding.vao))
        gl.glBindVertexArray(binding.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, buffer)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, buffer)
        self._attribute(0, vb, 'positions', base, 3)
        self._attribute(1, vb, 'uv', base, 2)
        self._attribute(2, vb, 'lightmap_uv', base, 2)
        self._attribute(3, colors, 'colors', color_base, 4, True)
        return binding

    def _attribute(self, location, buffer, name, base, components, integer=False):
        field = name + '_offset'
        if field not in buffer:
            gl.glDisableVertexAttribArray(location)
            if integer:
                gl.glVertexAttribI4ui(location, 255, 255, 255, 255)
            else:
                gl.glVertexAttrib2f(location, 0, 0)
            return
        gl.glEnableVertexAttribArray(location)
        pointer = ctypes.c_void_p(buffer[field] + base*components*4)
        if integer:
            gl.glVertexAttribIPointer(location, components, gl.GL_UNSIGNED_INT, 0, pointer)
        else:
            gl.glVertexAttribPointer(location, components, gl.GL_FLOAT, gl.GL_FALSE, 0, pointer)

    def _draw_mesh(self, entry):
        mesh, mat, albedo, lightmap, flip_a, flip_l, matrix, _, binding, mesh_color = entry
        buffers = self.gpu_buffers
        if binding is not self._bound_geometry:
            gl.glBindVertexArray(binding.vao)
            # Disabled attributes use context-global values, not VAO state.
            # Restore them after draws that may have enabled these attributes.
            if not binding.has_uv:
                gl.glVertexAttrib2f(1, 0, 0)
            if not binding.has_lightmap_uv:
                gl.glVertexAttrib2f(2, 0, 0)
            if not binding.has_colors:
                gl.glVertexAttribI4ui(3, 255, 255, 255, 255)
            self._bound_geometry = binding
        loc = self.locations
        model = mesh.get('matrix', IDENTITY)
        if model != self._bound_model:
            gl.glUniformMatrix4fv(loc['model'], 1, gl.GL_TRUE, matrix)
            self._bound_model = model
        if self.mode == 4:
            gl.glUniform3f(loc['mesh_color'], *mesh_color)
        use_lm = mesh['lightmap_index'] >= 0 and not mat.self_illumination
        use_colors = not mat.self_illumination and (not use_lm or mat.vertex_colors)
        has_colors = binding.has_colors
        material_state = (mat, albedo.id, lightmap.id, flip_a, flip_l, use_lm, use_colors, has_colors)
        if material_state != self._bound_material:
            for name, value in [('flip_albedo', flip_a), ('flip_lightmap', flip_l),
                                ('use_lightmap', use_lm), ('use_colors', use_colors),
                                ('has_colors', has_colors)]:
                gl.glUniform1i(loc[name], int(value))
            gl.glUniform3f(loc['tint'], *mat.color)
            gl.glUniform1f(loc['alpha_cutoff'], mat.alpha_cutoff)
            gain = 1. if mat.self_illumination else (2. if use_lm or mat.overbright else 1.)
            gl.glUniform1f(loc['gain'], gain)
            albedo.bind(0)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE if mat.clamp else gl.GL_REPEAT)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE if mat.clamp else gl.GL_REPEAT)
            lightmap.bind(1)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
            (gl.glEnable if mat.cull else gl.glDisable)(gl.GL_CULL_FACE)
            # Mesh colors show solid geometry, including non-depth-writing effects.
            (gl.glEnable if mat.depth_test or self.mode == 4 else gl.glDisable)(gl.GL_DEPTH_TEST)
            gl.glDepthMask(gl.GL_TRUE if mat.depth_write or self.mode == 4 else gl.GL_FALSE)
            (gl.glEnable if mat.depth_offset else gl.glDisable)(gl.GL_POLYGON_OFFSET_FILL)
            if mat.depth_offset:
                gl.glPolygonOffset(-1., -1.)
            if mat.blend and self.mode == 0:
                gl.glEnable(gl.GL_BLEND)
                source, dest = {'blend': (gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA),
                                'add': (gl.GL_ONE, gl.GL_ONE), 'modulate': (gl.GL_DST_COLOR, gl.GL_ZERO)}.get(mat.blend, (gl.GL_ONE, gl.GL_ZERO))
                gl.glBlendFunc(source, dest)
            else:
                gl.glDisable(gl.GL_BLEND)
            self._bound_material = material_state
        for prefix, primitive in [('', gl.GL_TRIANGLES), ('strip_', gl.GL_TRIANGLE_STRIP)]:
            count = mesh[prefix+'index_count']
            if count:
                ib = buffers[mesh[prefix+'index_buffer']]
                pointer = ctypes.c_void_p(ib['indices_offset'] + mesh[prefix+'index_start']*4)
                gl.glDrawElements(primitive, count, gl.GL_UNSIGNED_INT, pointer)

    def on_draw(self):
        gl.glDepthMask(gl.GL_TRUE)
        self.clear()
        if self.scene:
            fw, fh = self.get_framebuffer_size()
            scale = fh/max(self.height, 1)
            width, height = self.width, max(1, self.height-180)
            gl.glViewport(0, round(80*scale), fw, max(1, round(height*scale)))
            eye, view = self._camera()
            near, far = _map_clip_planes(eye, _camera_forward(self.yaw, self.pitch),
                                         self.scene.bounds_min, self.scene.bounds_max,
                                         draw_distance=self.draw_distance)
            projection = Mat4.perspective_projection(width/height, near, far, fov=50)
            mvp = projection @ view
            visible = visible_mesh_indices(self.scene.metadata, mvp)
            self.visible_mesh_count = len(visible)
            with self.program:
                self.program['mvp'] = mvp
                self.program['albedo'] = 0
                self.program['lightmap'] = 1
                gl.glUniform1i(self.locations['mode'], self.mode)
                gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE if self.wireframe else gl.GL_FILL)
                gl.glFrontFace(gl.GL_CCW)
                self._bound_geometry = self._bound_model = self._bound_material = None
                def opaque_order(entry):
                    mesh, material = entry[0], entry[1]
                    return (material.sort, _mesh_distance_squared(mesh, eye),
                            material.name, mesh['vertex_buffer'])
                for entry in sorted((entry for entry in self.opaque if entry[7] in visible),
                                    key=opaque_order):
                    self._draw_mesh(entry)
                def transparent_order(entry):
                    return entry[1].sort, -_mesh_distance_squared(entry[0], eye)
                for entry in sorted((entry for entry in self.transparent if entry[7] in visible),
                                    key=transparent_order):
                    self._draw_mesh(entry)
                gl.glBindVertexArray(0)
                gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
            gl.glViewport(0, 0, fw, fh)
        gl.glDepthMask(gl.GL_TRUE)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_CULL_FACE)
        gl.glDisable(gl.GL_POLYGON_OFFSET_FILL)
        gl.glDisable(gl.GL_BLEND)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        self.mode_label.text = f'{MODES[self.mode]} | {"Fly" if self.fly else "Orbit"} | speed {self.speed:.0f}'
        if self.scene:
            self.mode_label.text += (f' | {self.visible_mesh_count:,}/'
                                     f'{len(self.scene.metadata["meshes"]):,} visible')
        distance = 'Auto' if self.draw_distance is None else f'{self.draw_distance:,.1f}'
        self.distance_label.text = (f'Draw distance: {distance} | '
                                    'PgUp/PgDn increase/decrease | End auto')
        self.ui.draw()
        self.invalid = False

    def draw(self, dt):
        if self.invalid:
            super().draw(dt)

    def update(self, dt):
        if not self.fly or not self.scene:
            return
        forward = _camera_forward(self.yaw, self.pitch)
        right = forward.cross(Vec3(0, 0, 1)).normalize()
        movement = (forward*(self.keys[key.W]-self.keys[key.S]) + right*(self.keys[key.D]-self.keys[key.A])
                    + Vec3(0, 0, self.keys[key.E]-self.keys[key.Q]))
        if movement.length() > 0:
            self.eye += movement.normalize()*self.speed*min(dt, .1)*(5 if self.keys[key.LSHIFT] else 1)
            self.invalid = True

    def on_resize(self, width, height):
        if hasattr(self, 'ui'):
            self._layout()

    def on_expose(self):
        self.invalid = True

    def on_scale(self, scale, dpi):
        self.invalid = True

    def on_mouse_press(self, x, y, button, modifiers):
        if not self._camera_buttons and button == mouse.LEFT:
            for rect, _, action in self.buttons:
                if rect.x <= x <= rect.x+rect.width and rect.y <= y <= rect.y+rect.height:
                    action()
                    return
        if button not in (mouse.LEFT, mouse.MIDDLE, mouse.RIGHT):
            return
        if not self._camera_buttons:
            if self.scene is None or not 80 < y < self.height-100:
                return
            self.set_exclusive_mouse(True)
        self._camera_buttons |= button
        if button == mouse.RIGHT:
            self._set_fly(True)

    def _release_camera_mouse(self):
        self._camera_buttons = 0
        self.set_exclusive_mouse(False)

    def on_mouse_release(self, x, y, button, modifiers):
        self._camera_buttons &= ~button
        if not self._camera_buttons:
            self._release_camera_mouse()

    def on_deactivate(self):
        self._release_camera_mouse()

    def on_mouse_motion(self, x, y, dx, dy):
        # Win32 raw input can report motion rather than drag after capture.
        if self._camera_buttons:
            self.on_mouse_drag(x, y, dx, dy, self._camera_buttons, 0)

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if not self._camera_buttons:
            return
        buttons &= self._camera_buttons
        if buttons & mouse.RIGHT or buttons & mouse.LEFT:
            direction = 1 if self.fly else -1
            self.yaw -= direction*dx*.005
            self.pitch = max(-1.55, min(1.55, self.pitch-direction*dy*.005))
        elif buttons & mouse.MIDDLE:
            right = Vec3(-math.sin(self.yaw), math.cos(self.yaw), 0)
            up = Vec3(0, 0, 1)
            delta = (right*(-dx)+up*(-dy))*self.distance/max(self.height-180, 1)
            if self.fly:
                self.eye += delta
            else:
                self.target += delta
        self.invalid = True

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        if self.fly:
            self.speed = max(.1, min(1e8, self.speed*math.exp(scroll_y*.15)))
        else:
            self.distance = max(.05, min(self.radius*100, self.distance*math.exp(-scroll_y*.15)))
        self.invalid = True

    def on_key_press(self, symbol, modifiers):
        if symbol == key.O and modifiers & key.MOD_CTRL:
            self.open_file()
        elif symbol in (key._1, key._2, key._3, key._4, key._5):
            self.mode = (key._1, key._2, key._3, key._4, key._5).index(symbol)
        elif symbol == key.HOME:
            self.frame_map()
        elif symbol == key.F:
            self._set_fly(not self.fly)
        elif symbol == key.G:
            self.wireframe = not self.wireframe
        elif symbol == key.PAGEUP:
            self._adjust_draw_distance(1.25)
        elif symbol == key.PAGEDOWN:
            self._adjust_draw_distance(.8)
        elif symbol == key.END:
            self.draw_distance = None
        elif symbol in (key.BRACKETLEFT, key.BRACKETRIGHT):
            self.change_map(-1 if symbol == key.BRACKETLEFT else 1)
        elif symbol == key.ESCAPE:
            self.close()
        self.invalid = True

    def open_file(self):
        self._release_camera_mouse()
        try:
            directory = self.scene.path.parent if self.scene else Path.cwd()
            path = choose_file(getattr(self, '_hwnd', None), directory, extension='ds2',
                               description='DS2 maps', title='Open DS2 map')
            if path:
                self.queue_load(path)
        except OSError as error:
            self._status(str(error), True)
        self.invalid = True

    def change_map(self, delta):
        if self.map_paths:
            current = self.scene.path if self.scene else None
            index = self.map_paths.index(current) if current in self.map_paths else 0
            self.queue_load(self.map_paths[(index+delta) % len(self.map_paths)])

    def on_file_drop(self, x, y, paths):
        if paths:
            self.queue_load(next((p for p in paths if Path(p).suffix.casefold() == '.ds2'), paths[0]))

    def _capture(self, dt):
        self.switch_to()
        self.on_draw()
        pyglet.image.get_buffer_manager().get_color_buffer().save(str(self.screenshot))
        print(f'Saved {self.screenshot}')
        self.close()

    def close(self):
        self._release_camera_mouse()
        pyglet.clock.unschedule(self.update)
        pyglet.clock.unschedule(self._load_pending)
        pyglet.clock.unschedule(self._capture)
        self._release_scene()
        for texture in self.constants.values():
            texture.delete()
        self.program.delete()
        super().close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('map', nargs='?', type=Path)
    parser.add_argument('--mode', choices=('combined', 'albedo', 'lightmap', 'colors', 'meshes'), default='combined')
    parser.add_argument('--screenshot', type=Path, help='Save a rendered frame and exit')
    args = parser.parse_args()
    try:
        viewer = MapViewer(args.map, mode=('combined', 'albedo', 'lightmap', 'colors', 'meshes').index(args.mode), screenshot=args.screenshot)
        pyglet.app.run()
    except Exception as error:
        print(f'DS2 viewer: {error}', file=sys.stderr)
        return 1
    return int(viewer.load_error)


if __name__ == '__main__':
    raise SystemExit(main())
