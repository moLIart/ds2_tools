"""Local DS2MD albedo and animation viewer. Run: python view_ds2md.py [MODEL.ds2md]."""

from __future__ import annotations

import argparse
from array import array
import ctypes
import math
from pathlib import Path
import sys

import pyglet

pyglet.options['shadow_window'] = False

from pyglet import gl
from pyglet.graphics.shader import Shader, ShaderProgram
from pyglet.math import Mat4, Vec3
from pyglet.window import key, mouse

from ds2md_scene import load_scene
from ds2md_animation import Animator


MESH_VERTEX = """#version 330 core
in vec3 position;
in vec3 normal;
in vec2 uv_coords;
in uvec2 influence_range;
uniform sampler2D influence_buffer;
uniform sampler2D bone_matrices;
uniform mat4 mvp;
out vec3 surface_normal;
out vec2 texture_uv;
vec4 fetch_data(sampler2D data_texture, int index) {
    int width = textureSize(data_texture, 0).x;
    return texelFetch(data_texture, ivec2(index % width, index / width), 0);
}
void main() {
    vec4 animated_position = vec4(position, 1.0);
    vec3 animated_normal = normal;
    if (influence_range.y != 0u) {
        mat4 skin = mat4(0.0);
        for (uint i = 0u; i < influence_range.y; ++i) {
            vec2 influence = fetch_data(influence_buffer, int(influence_range.x + i)).rg;
            int offset = int(influence.x) * 4;
            mat4 bone = mat4(fetch_data(bone_matrices, offset),
                             fetch_data(bone_matrices, offset + 1),
                             fetch_data(bone_matrices, offset + 2),
                             fetch_data(bone_matrices, offset + 3));
            skin += bone * influence.y;
        }
        animated_position = vec4((skin * vec4(position, 1.0)).xyz, 1.0);
        animated_normal = mat3(skin) * normal;
    }
    gl_Position = mvp * animated_position;
    surface_normal = animated_normal;
    texture_uv = uv_coords;
}
"""
MESH_FRAGMENT = """#version 330 core
in vec3 surface_normal;
in vec2 texture_uv;
uniform vec3 base_color;
uniform vec3 light_direction; // Unit vector toward the light in model/world space.
uniform sampler2D albedo_texture;
uniform bool has_albedo;
uniform bool flip_v;
out vec4 final_color;
void main() {
    float magnitude = length(surface_normal);
    vec3 n = magnitude > 0.00001 ? surface_normal / magnitude : vec3(0.0, 0.0, 1.0);
    if (!gl_FrontFacing) n = -n;
    float diffuse = max(dot(n, light_direction), 0.0);
    vec2 uv = vec2(texture_uv.x, flip_v ? 1.0 - texture_uv.y : texture_uv.y);
    vec3 color = base_color;
    if (has_albedo) color *= texture(albedo_texture, uv).rgb;
    final_color = vec4(color * (0.30 + 0.70 * diffuse), 1.0);
}
"""
LINE_VERTEX = """#version 330 core
in vec3 position;
uniform mat4 mvp;
void main() { gl_Position = mvp * vec4(position, 1.0); }
"""
LINE_FRAGMENT = """#version 330 core
uniform vec3 line_color;
out vec4 final_color;
void main() { final_color = vec4(line_color, 1.0); }
"""
PALETTE = ((0.76, 0.79, 0.82), (0.68, 0.73, 0.77), (0.78, 0.74, 0.68),
           (0.66, 0.73, 0.71), (0.74, 0.70, 0.75))


def _clip_planes(eye, target, distance, bounds_min, bounds_max):
    """Fit visible depth, with a stable near floor when the eye enters bounds.

    distance is the positive orbit distance. Do not collapse near to an epsilon
    when an animated bounding volume crosses the eye: that destroys 24-bit depth
    precision and causes temporal surface fighting.
    """
    forward = tuple((target[axis] - eye[axis]) / distance for axis in range(3))
    center_depth = sum(((low + high) * 0.5 - eye[axis]) * forward[axis]
                       for axis, (low, high) in enumerate(zip(bounds_min, bounds_max)))
    half_depth = sum((high - low) * 0.5 * abs(forward[axis])
                     for axis, (low, high) in enumerate(zip(bounds_min, bounds_max)))
    padding = max(half_depth * 0.1, distance * 0.001)
    near = max(distance * 0.01, center_depth - half_depth - padding)
    far = max(near + padding, center_depth + half_depth + padding)
    return near, far


class FloatTexture:
    """Packed RG/RGBA float data using pyglet-supported sampler2D uniforms."""

    def __init__(self, values, internal_format):
        self.texture = gl.GLuint()
        self.internal_format = internal_format
        self.channels = 2 if internal_format == gl.GL_RG32F else 4
        self.format = gl.GL_RG if self.channels == 2 else gl.GL_RGBA
        self.width = self.height = 0
        maximum = gl.GLint()
        gl.glGetIntegerv(gl.GL_MAX_TEXTURE_SIZE, ctypes.byref(maximum))
        self.max_size = maximum.value
        try:
            gl.glGenTextures(1, ctypes.byref(self.texture))
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)
            for parameter in (gl.GL_TEXTURE_MIN_FILTER, gl.GL_TEXTURE_MAG_FILTER):
                gl.glTexParameteri(gl.GL_TEXTURE_2D, parameter, gl.GL_NEAREST)
            for parameter in (gl.GL_TEXTURE_WRAP_S, gl.GL_TEXTURE_WRAP_T):
                gl.glTexParameteri(gl.GL_TEXTURE_2D, parameter, gl.GL_CLAMP_TO_EDGE)
            self.upload(values)
        except Exception:
            self.delete()
            raise
        finally:
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    def upload(self, values):
        values = values if isinstance(values, array) else array('f', values)
        count = len(values) // self.channels
        if not count or len(values) % self.channels:
            raise ValueError('Float texture data must contain complete, nonempty texels.')
        width = min(count, 1024, self.max_size)
        height = (count + width - 1) // width
        if height > self.max_size:
            raise ValueError('Skeletal data exceeds the GPU texture-size limit.')
        pointer = (gl.GLfloat * len(values)).from_buffer(values)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)
        if (width, height) != (self.width, self.height):
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, self.internal_format, width, height,
                            0, self.format, gl.GL_FLOAT, None)
            self.width, self.height = width, height
        rows, remainder = divmod(count, width)
        if rows:
            gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, width, rows,
                               self.format, gl.GL_FLOAT, pointer)
        if remainder:
            offset = rows * width * self.channels * values.itemsize
            gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, rows, remainder, 1,
                               self.format, gl.GL_FLOAT, ctypes.byref(pointer, offset))
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    def bind(self, unit):
        gl.glActiveTexture(gl.GL_TEXTURE0 + unit)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)

    def delete(self):
        if self.texture.value:
            gl.glDeleteTextures(1, ctypes.byref(self.texture))
            self.texture.value = 0


class AnimatedBounds:
    """Conservative skin bounds; the per-pose work depends only on bone count."""

    def __init__(self, scene, bone_count):
        self.bones = [None] * bone_count
        self.unweighted = None
        self.weight_min = math.inf
        self.weight_max = 0.0
        for mesh in scene.meshes:
            for vertex in range(len(mesh.positions) // 3):
                point = mesh.positions[vertex * 3:vertex * 3 + 3]
                start, count = mesh.influence_ranges[vertex * 2:vertex * 2 + 2]
                if not bone_count or not count:
                    self.unweighted = self._include(self.unweighted, point)
                    continue
                total = 0.0
                for entry in range(start, start + count):
                    bone, weight = mesh.influences[entry * 2:entry * 2 + 2]
                    total += weight
                    self.bones[int(bone)] = self._include(self.bones[int(bone)], point)
                self.weight_min = min(self.weight_min, total)
                self.weight_max = max(self.weight_max, total)
        self.fallback = (scene.bounds_min, scene.bounds_max)

    @staticmethod
    def _include(bounds, point):
        if bounds is None:
            return (list(point), list(point))
        low, high = bounds
        for axis in range(3):
            low[axis] = min(low[axis], point[axis])
            high[axis] = max(high[axis], point[axis])
        return bounds

    def evaluate(self, matrices):
        weighted = None
        for bone, bounds in enumerate(self.bones):
            if bounds is None:
                continue
            low, high = bounds
            offset = bone * 16
            for x in (low[0], high[0]):
                for y in (low[1], high[1]):
                    for z in (low[2], high[2]):
                        point = tuple(matrices[offset + row] * x
                                      + matrices[offset + 4 + row] * y
                                      + matrices[offset + 8 + row] * z
                                      + matrices[offset + 12 + row] for row in range(3))
                        weighted = self._include(weighted, point)
        # A raw weighted sum is a convex combination times its total weight.
        # Rescale the union for all observed totals, rather than assuming sum=1.
        result = None
        if weighted is not None:
            low, high = weighted
            candidates = [(low[axis] * self.weight_min, low[axis] * self.weight_max,
                           high[axis] * self.weight_min, high[axis] * self.weight_max)
                          for axis in range(3)]
            result = ([min(values) for values in candidates],
                      [max(values) for values in candidates])
        if self.unweighted is not None:
            for point in self.unweighted:
                result = self._include(result, point)
        return result if result is not None else self.fallback


def load_albedo_texture(path: Path) -> pyglet.image.Texture:
    """Decode and upload one repeating albedo; requires a current GL context."""
    if path.suffix.casefold() == '.tga':
        # Windows WIC does not decode the game's TGA; use pyglet's Pillow codec.
        from pyglet.image.codecs.pil import PILImageDecoder
        decoder = PILImageDecoder()
    else:
        # Keep DDS compressed and unflipped even when other codecs are installed.
        from pyglet.image.codecs.dds import DDSImageDecoder
        decoder = DDSImageDecoder()
    image = pyglet.image.load(str(path), decoder=decoder)
    texture = pyglet.image.Texture.create(
        image.width, image.height, internalformat=None,
        min_filter=gl.GL_LINEAR, mag_filter=gl.GL_LINEAR)
    try:
        texture.bind()
        gl.glTexParameteri(texture.target, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
        gl.glTexParameteri(texture.target, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
        if isinstance(image, pyglet.image.CompressedImageData):
            if image.extension and not gl.gl_info.have_extension(image.extension):
                raise pyglet.image.ImageException(f'{image.extension} is required for {path.name}')
            gl.glCompressedTexImage2D(
                texture.target, 0, image.gl_format, image.width, image.height,
                0, len(image.data), image.data)
        else:
            image.blit_to_texture(texture.target, 0, 0, 0, 0, internalformat=gl.GL_RGBA)
    except Exception:
        texture.delete()
        raise
    finally:
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
    return texture


def choose_file(owner, initial_directory: Path) -> Path | None:
    """Use the Windows common dialog without tkinter or changing the cwd."""
    if sys.platform != 'win32':
        raise OSError('Native Open is Windows-only; use a CLI path or drop a model file.')
    from ctypes import wintypes

    class OPENFILENAMEW(ctypes.Structure):
        _fields_ = [
            ('lStructSize', wintypes.DWORD), ('hwndOwner', wintypes.HWND),
            ('hInstance', wintypes.HINSTANCE), ('lpstrFilter', wintypes.LPCWSTR),
            ('lpstrCustomFilter', wintypes.LPWSTR), ('nMaxCustFilter', wintypes.DWORD),
            ('nFilterIndex', wintypes.DWORD), ('lpstrFile', wintypes.LPWSTR),
            ('nMaxFile', wintypes.DWORD), ('lpstrFileTitle', wintypes.LPWSTR),
            ('nMaxFileTitle', wintypes.DWORD), ('lpstrInitialDir', wintypes.LPCWSTR),
            ('lpstrTitle', wintypes.LPCWSTR), ('Flags', wintypes.DWORD),
            ('nFileOffset', wintypes.WORD), ('nFileExtension', wintypes.WORD),
            ('lpstrDefExt', wintypes.LPCWSTR), ('lCustData', wintypes.LPARAM),
            ('lpfnHook', ctypes.c_void_p), ('lpTemplateName', wintypes.LPCWSTR),
            ('pvReserved', ctypes.c_void_p), ('dwReserved', wintypes.DWORD),
            ('FlagsEx', wintypes.DWORD),
        ]

    library = ctypes.WinDLL('comdlg32', use_last_error=True)
    library.GetOpenFileNameW.argtypes = [ctypes.POINTER(OPENFILENAMEW)]
    library.GetOpenFileNameW.restype = wintypes.BOOL
    library.CommDlgExtendedError.argtypes = []
    library.CommDlgExtendedError.restype = wintypes.DWORD
    buffer = ctypes.create_unicode_buffer(32768)
    options = OPENFILENAMEW()
    options.lStructSize = ctypes.sizeof(options)
    options.hwndOwner = owner
    options.lpstrFilter = 'DS2MD models (*.ds2md)\0*.ds2md\0All files (*.*)\0*.*\0\0'
    options.nFilterIndex = 1
    options.lpstrFile = ctypes.cast(buffer, wintypes.LPWSTR)
    options.nMaxFile = len(buffer)
    options.lpstrInitialDir = str(initial_directory)
    options.lpstrTitle = 'Open DS2MD model'
    # EXPLORER | FILEMUSTEXIST | PATHMUSTEXIST | NOCHANGEDIR
    options.Flags = 0x00080000 | 0x00001000 | 0x00000800 | 0x00000008
    if library.GetOpenFileNameW(ctypes.byref(options)):
        return Path(buffer.value)
    error = library.CommDlgExtendedError()
    if error:
        raise OSError(f'Windows file dialog failed (0x{error:08X}).')
    return None


class ModelViewer(pyglet.window.Window):
    def __init__(self, initial_path: Path | None = None):
        config = gl.Config(double_buffer=True, depth_size=24, major_version=3, minor_version=3)
        super().__init__(width=1200, height=800, caption='DS2MD Geometry Viewer',
                         resizable=True, config=config, file_drops=True, vsync=True)
        self.set_minimum_size(640, 420)
        self.scene = None
        self.mesh_lists = []
        self.albedo_textures = {}
        self.grid_list = None
        self.grid_bounds = None
        self.animator = None
        self.palette_buffer = None
        self.bounds_cache = None
        self.bounds_min = (0, 0, 0)
        self.bounds_max = (0, 0, 0)
        self._clock_scheduled = False
        self._timeline_drag = False
        self._ui_capture = False
        self._resume_after_seek = False
        self.pending_path = None
        self.wireframe = False
        self.show_grid = True
        self.target = Vec3(0, 0, 0)
        self.radius = 1.0
        self.distance = 4.0
        self.yaw = math.radians(-55)
        self.pitch = math.radians(20)
        self.reset_light()
        self.mesh_program = ShaderProgram(Shader(MESH_VERTEX, 'vertex'), Shader(MESH_FRAGMENT, 'fragment'))
        self.line_program = ShaderProgram(Shader(LINE_VERTEX, 'vertex'), Shader(LINE_FRAGMENT, 'fragment'))
        self.ui_batch = pyglet.graphics.Batch()
        self.open_button = pyglet.shapes.Rectangle(12, 0, 88, 30, color=(52, 82, 110), batch=self.ui_batch)
        self.open_label = pyglet.text.Label('Open...', font_size=11, x=56, anchor_x='center',
                                           anchor_y='center', batch=self.ui_batch)
        self.title_label = pyglet.text.Label('No model loaded', font_size=13, x=114,
                                            anchor_y='top', batch=self.ui_batch)
        self.stats_label = pyglet.text.Label('', font_size=9, x=14, anchor_y='top',
                                            multiline=True, width=1100, batch=self.ui_batch)
        self.note_label = pyglet.text.Label('', font_size=10, x=14, anchor_y='top',
                                           multiline=False, width=1100, batch=self.ui_batch,
                                           color=(180, 190, 202, 255))
        self.controls_label = pyglet.text.Label('', font_size=9, x=14, y=67,
                                                anchor_y='top', multiline=True,
                                                width=1100, batch=self.ui_batch)
        self.status_label = pyglet.text.Label('Open or drop a DS2MD model to begin.', font_size=9,
                                             x=14, y=12, batch=self.ui_batch)
        self.clip_label = pyglet.text.Label('Bind pose (no animation)', font_size=10,
                                           x=14, y=153, batch=self.ui_batch)
        self.time_label = pyglet.text.Label('0.00 / 0.00 s', font_size=9,
                                           anchor_x='right', y=88, batch=self.ui_batch)
        self.timeline = pyglet.shapes.Rectangle(14, 90, 400, 8, color=(67, 79, 96),
                                                batch=self.ui_batch)
        self.timeline_fill = pyglet.shapes.Rectangle(14, 90, 0, 8, color=(104, 164, 210),
                                                     batch=self.ui_batch)
        self.timeline_knob = pyglet.shapes.Circle(14, 94, 5, color=(190, 218, 238),
                                                 batch=self.ui_batch)
        self.animation_buttons = {}
        for name, text, x, width in (
                ('previous', '< [', 14, 44), ('next', '] >', 64, 44),
                ('play', 'Play', 114, 64), ('bind', 'Bind', 184, 50),
                ('slower', '-', 240, 30), ('faster', '+', 340, 30),
                ('repeat', 'Repeat: on', 380, 100)):
            rectangle = pyglet.shapes.Rectangle(x, 111, width, 28, color=(52, 69, 88),
                                                batch=self.ui_batch)
            label = pyglet.text.Label(text, font_size=10, x=x + width / 2, y=125,
                                      anchor_x='center', anchor_y='center', batch=self.ui_batch)
            self.animation_buttons[name] = (rectangle, label)
        self.speed_label = pyglet.text.Label('1.00x', font_size=10, x=305, y=125,
                                            anchor_x='center', anchor_y='center',
                                            batch=self.ui_batch)
        self.empty_label = pyglet.text.Label('Drop a .ds2md file here\nor click Open (Ctrl+O)',
                                            font_size=18, anchor_x='center', anchor_y='center',
                                            align='center', multiline=True, width=520,
                                            batch=self.ui_batch, color=(175, 187, 201, 255))
        self._layout()
        self._update_controls()
        self._update_animation_ui()
        gl.glClearColor(0.075, 0.09, 0.115, 1.0)
        gl.glDisable(gl.GL_CULL_FACE)
        if initial_path is not None:
            self.queue_load(initial_path)

    def _layout(self):
        self.open_button.y = self.height - 44
        self.open_label.y = self.height - 29
        self.title_label.y = self.height - 18
        self.title_label.width = max(100, self.width - 130)
        self.stats_label.y = self.height - 54
        self.stats_label.width = max(100, self.width - 28)
        self.note_label.y = self.height - 88
        self.note_label.width = max(100, self.width - 28)
        self.status_label.width = max(100, self.width - 28)
        self.controls_label.width = max(100, self.width - 28)
        self.time_label.x = self.width - 14
        self.timeline.width = max(100, self.width - 188)
        self._update_animation_ui()
        if self.scene is not None:
            self._set_label(self.title_label, f'{self.scene.path.name}  |  {self.scene.name}',
                            self.width - 130)
            self._set_label(self.note_label, self.scene.note, self.width - 28)
        self.empty_label.x = self.width / 2
        self.empty_label.y = self.height / 2
        self.invalid = True

    def _update_controls(self):
        self.controls_label.text = (
            'LMB orbit | Shift+LMB light | RMB/MMB pan | Wheel zoom | Ctrl+O open\n'
            f'F frame | L light | W wire: {"on" if self.wireframe else "off"} | '
            f'G grid: {"on" if self.show_grid else "off"} | Space play | Arrows seek | Home start\n'
            '[ / ] clips | B bind | - / + speed | R repeat')
        self.invalid = True

    def _status(self, text, error=False):
        self._set_label(self.status_label, text, self.width - 28)
        self.status_label.color = (255, 155, 145, 255) if error else (190, 205, 220, 255)
        self.invalid = True

    @staticmethod
    def _set_label(label, text, width):
        """Fit single-line metadata without letting long filenames cover controls."""
        text = ' '.join(str(text).split())
        label.text = text
        if label.content_width > width:
            low, high = 0, len(text)
            while low < high:
                middle = (low + high + 1) // 2
                label.text = text[:middle] + '...'
                if label.content_width <= width:
                    low = middle
                else:
                    high = middle - 1
            label.text = text[:low] + '...'

    def _update_animation_ui(self):
        animator = self.animator
        if animator is not None and animator.clip_index is not None:
            clip = animator.skeleton.clips[animator.clip_index]
            name = f'{animator.clip_index + 1}/{len(animator.skeleton.clips)}  {clip.name}'
        else:
            name = 'Bind pose' if animator is not None else 'Static model / no animation'
        self._set_label(self.clip_label, name, self.width - 28)
        time = animator.time_seconds if animator is not None else 0.0
        duration = animator.duration_seconds if animator is not None else 0.0
        fraction = min(1.0, max(0.0, time / duration)) if duration else 0.0
        self.timeline_fill.width = self.timeline.width * fraction
        self.timeline_knob.x = self.timeline.x + self.timeline.width * fraction
        self.time_label.text = f'{time:.2f} / {duration:.2f} s'
        self.animation_buttons['play'][1].text = 'Pause' if animator and animator.playing else 'Play'
        self.animation_buttons['repeat'][1].text = (
            f'Repeat: {"on" if animator.loop else "off"}' if animator else 'Repeat: on')
        self.speed_label.text = f'{animator.speed if animator else 1.0:.2f}x'

    def _sync_clock(self):
        active = bool(self.animator and self.animator.playing and
                      self.animator.clip_index is not None and not self._timeline_drag)
        if active and not self._clock_scheduled:
            pyglet.clock.schedule_interval(self.update_animation, 1 / 60)
            self._clock_scheduled = True
        elif not active and self._clock_scheduled:
            pyglet.clock.unschedule(self.update_animation)
            self._clock_scheduled = False

    def _pose_changed(self):
        if self.animator is not None and self.animator.matrices:
            self.switch_to()
            self.palette_buffer.upload(self.animator.matrices)
        self.bounds_min, self.bounds_max = self.bounds_cache.evaluate(
            self.animator.matrices if self.animator is not None else ())
        self.radius = max(math.sqrt(sum((b - a) ** 2 for a, b in
                                       zip(self.bounds_min, self.bounds_max))) * 0.5, 0.001)
        self._update_animation_ui()
        self.invalid = True

    def update_animation(self, dt):
        """Clock callback; evaluate bones and upload only when animation advances."""
        was_playing = bool(self.animator and self.animator.playing)
        if self.animator is not None and self.animator.update(dt):
            self._pose_changed()
        self._sync_clock()
        self._update_animation_ui()
        if was_playing != bool(self.animator and self.animator.playing):
            self.invalid = True

    def select_clip(self, index):
        """Select a clip index, or None for bind pose; frame only on selection."""
        if self.animator is None:
            return
        try:
            self.animator.select_clip(index)
        except (ValueError, IndexError) as exc:
            self._status(f'Cannot select animation: {exc}', error=True)
            return
        self._pose_changed()
        self.frame_model()
        self._sync_clock()

    def cycle_clip(self, direction):
        if self.animator is None:
            return
        count = len(self.animator.skeleton.clips)
        current = self.animator.clip_index
        selection = ((0 if current is None else current + 1) + direction) % (count + 1)
        self.select_clip(None if selection == 0 else selection - 1)

    def toggle_playback(self):
        if self.animator is None:
            return
        if self.animator.clip_index is None:
            if self.animator.skeleton.clips:
                self.select_clip(0)
            return
        if not self.animator.playing and self.animator.time_seconds >= self.animator.duration_seconds:
            self.seek_animation(0)
        self.animator.playing = not self.animator.playing
        self._sync_clock()
        self._update_animation_ui()
        self.invalid = True

    def seek_animation(self, seconds):
        """Seek without reframing or changing the existing play/pause state."""
        if self.animator is not None and self.animator.clip_index is not None:
            self.animator.seek(seconds)
            self._pose_changed()
            self._sync_clock()

    def set_animation_speed(self, speed):
        if self.animator is not None:
            self.animator.speed = max(0.125, min(8.0, speed))
            self._update_animation_ui()
            self.invalid = True

    def toggle_repeat(self):
        if self.animator is not None:
            self.animator.loop = not self.animator.loop
            self._update_animation_ui()
            self.invalid = True

    def _seek_timeline(self, x):
        if self.animator is not None:
            fraction = max(0.0, min(1.0, (x - self.timeline.x) / self.timeline.width))
            self.seek_animation(fraction * self.animator.duration_seconds)

    def queue_load(self, path):
        self.pending_path = Path(path).expanduser()
        self._status(f'Loading {self.pending_path.name} ...')
        pyglet.clock.unschedule(self._load_pending)
        # Let the event loop paint the loading message before synchronous disk/GPU work.
        pyglet.clock.schedule_once(self._load_pending, 0.05)

    def _load_pending(self, dt):
        path = self.pending_path
        self.pending_path = None
        if path is None:
            return
        self.switch_to()
        allocated = []
        textures = {}
        buffers = []
        try:
            scene = load_scene(path)
            animator = Animator(scene.skeleton) if scene.skeleton is not None else None
            if animator is not None and scene.skeleton.clips:
                animator.select_clip(0)
            identity = array('f', (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1))
            palette = FloatTexture(animator.matrices if animator and animator.matrices else identity,
                                   gl.GL_RGBA32F)
            buffers.append(palette)
            bone_count = len(scene.skeleton.bones) if scene.skeleton is not None else 0
            bounds_cache = AnimatedBounds(scene, bone_count)
            low, high = bounds_cache.evaluate(animator.matrices if animator else ())
            meshes = []
            warnings = list(scene.texture_warnings)
            for mesh in scene.meshes:
                if not mesh.indices:
                    continue
                texture = None
                if mesh.albedo_path is not None:
                    if mesh.albedo_path not in textures:
                        try:
                            textures[mesh.albedo_path] = load_albedo_texture(mesh.albedo_path)
                        except Exception as exc:
                            textures[mesh.albedo_path] = None
                            warnings.append(f'Albedo {mesh.albedo_path.name}: {exc}')
                    texture = textures[mesh.albedo_path]
                influence_buffer = FloatTexture(mesh.influences or (0.0, 0.0), gl.GL_RG32F)
                buffers.append(influence_buffer)
                ranges = mesh.influence_ranges if bone_count else (0, 0) * (len(mesh.positions) // 3)
                vertex_list = self.mesh_program.vertex_list_indexed(
                    len(mesh.positions) // 3, gl.GL_TRIANGLES, mesh.indices,
                    position=('f', mesh.positions), normal=('f', mesh.normals),
                    uv_coords=('f', mesh.uv_coords), influence_range=('I', ranges))
                allocated.append(vertex_list)
                meshes.append((vertex_list, texture,
                               mesh.albedo_path is not None and
                               mesh.albedo_path.suffix.casefold() == '.tga', influence_buffer))
            center = Vec3(*((a + b) * 0.5 for a, b in zip(low, high)))
            radius = max(math.sqrt(sum((b - a) ** 2 for a, b in zip(low, high))) * 0.5, 0.001)
            extent = radius * 1.5
            z = low[2] - radius * 0.005
            grid_bounds = ((center.x - extent, center.y - extent, z),
                           (center.x + extent, center.y + extent, z))
            grid_vertices = []
            for step in range(-10, 11):
                offset = step * extent / 10
                grid_vertices.extend((center.x - extent, center.y + offset, z,
                                      center.x + extent, center.y + offset, z,
                                      center.x + offset, center.y - extent, z,
                                      center.x + offset, center.y + extent, z))
            grid = self.line_program.vertex_list(len(grid_vertices) // 3, gl.GL_LINES,
                                                 position=('f', grid_vertices))
            allocated.append(grid)
        except Exception as exc:
            for vertex_list in allocated:
                vertex_list.delete()
            for buffer in buffers:
                buffer.delete()
            for texture in textures.values():
                if texture is not None:
                    texture.delete()
            self._status(f'Cannot open {path.name}: {exc}', error=True)
            return

        self._release_scene_resources()
        self.scene = scene
        self.mesh_lists = meshes
        self.albedo_textures = textures
        self.grid_list = grid
        self.grid_bounds = grid_bounds
        self.animator = animator
        self.palette_buffer = palette
        self.bounds_cache = bounds_cache
        self.bounds_min, self.bounds_max = low, high
        self._timeline_drag = False
        self._ui_capture = False
        self.radius = radius
        self.frame_model()
        self.set_caption(f'{scene.path.name} - DS2MD Geometry Viewer')
        self._set_label(self.title_label, f'{scene.path.name}  |  {scene.name}', self.width - 130)
        vertices = sum(len(mesh.positions) // 3 for mesh in scene.meshes)
        triangles = sum(len(mesh.indices) // 3 for mesh in scene.meshes)
        textured_meshes = sum(texture is not None for _, texture, _, _ in meshes)
        texture_count = sum(texture is not None for texture in textures.values())
        clips = len(scene.skeleton.clips) if scene.skeleton is not None else 0
        self.stats_label.text = (f'DS2MD {scene.version} | {len(scene.meshes):,} meshes | '
                                 f'{vertices:,} vertices | {triangles:,} triangles\n'
                                 f'Albedo {textured_meshes}/{len(meshes)} meshes / '
                                 f'{texture_count} textures | {bone_count} bones | {clips} clips')
        self._set_label(self.note_label, scene.note, self.width - 28)
        self.empty_label.text = ''
        if warnings:
            for warning in warnings:
                print(f'{scene.path.name}: {warning}', file=sys.stderr)
            extra = f' (+{len(warnings) - 1} more; see console)' if len(warnings) > 1 else ''
            self._status(f'{scene.path} | Albedo warnings ({len(warnings)}): {warnings[0]}{extra}',
                         error=True)
        else:
            self._status(str(scene.path))
        self._update_controls()
        self._update_animation_ui()
        self._sync_clock()

    def _release_scene_resources(self):
        pyglet.clock.unschedule(self.update_animation)
        self._clock_scheduled = False
        for vertex_list, _, _, influence_buffer in self.mesh_lists:
            vertex_list.delete()
            influence_buffer.delete()
        self.mesh_lists.clear()
        for texture in self.albedo_textures.values():
            if texture is not None:
                texture.delete()
        self.albedo_textures.clear()
        if self.palette_buffer is not None:
            self.palette_buffer.delete()
            self.palette_buffer = None
        if self.grid_list is not None:
            self.grid_list.delete()
            self.grid_list = None
        self.grid_bounds = None

    def _model_viewport(self):
        # Reserve header and playback controls instead of drawing through them.
        return max(self.width, 1), max(self.height - 280, 1)

    def frame_model(self):
        if self.scene is not None:
            self.target = Vec3(*((a + b) * 0.5 for a, b in
                                  zip(self.bounds_min, self.bounds_max)))
        self.yaw = math.radians(-55)
        self.pitch = math.radians(20)
        half_fov = math.radians(45) / 2
        width, height = self._model_viewport()
        aspect = width / height
        limiting_angle = min(half_fov, math.atan(math.tan(half_fov) * aspect))
        self.distance = self.radius / math.sin(limiting_angle) * 1.25
        self.invalid = True

    def reset_light(self):
        self.light_yaw = math.atan2(-0.55, 0.35)
        self.light_pitch = math.atan2(0.85, math.hypot(0.35, -0.55))
        self.invalid = True

    def _light_direction(self):
        horizontal = math.cos(self.light_pitch)
        return (horizontal * math.cos(self.light_yaw),
                horizontal * math.sin(self.light_yaw), math.sin(self.light_pitch))

    def _camera(self):
        offset = Vec3(math.cos(self.pitch) * math.cos(self.yaw),
                      math.cos(self.pitch) * math.sin(self.yaw), math.sin(self.pitch))
        eye = self.target + offset * self.distance
        return eye, Mat4.look_at(eye, self.target, Vec3(0, 0, 1))

    def draw(self, dt):
        if self.invalid:
            super().draw(dt)

    def on_draw(self):
        self.clear()
        if self.scene is not None:
            width, height = self._model_viewport()
            framebuffer_width, framebuffer_height = self.get_framebuffer_size()
            pixel_scale = framebuffer_height / max(self.height, 1)
            gl.glViewport(0, round(174 * pixel_scale), framebuffer_width,
                          max(1, round(height * pixel_scale)))
            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glDepthFunc(gl.GL_LEQUAL)
            gl.glDisable(gl.GL_CULL_FACE)
            eye, view = self._camera()
            low, high = self.bounds_min, self.bounds_max
            if self.show_grid and self.grid_bounds is not None:
                low = tuple(min(a, b) for a, b in zip(low, self.grid_bounds[0]))
                high = tuple(max(a, b) for a, b in zip(high, self.grid_bounds[1]))
            near, far = _clip_planes(eye, self.target, self.distance, low, high)
            projection = Mat4.perspective_projection(width / height,
                                                    near, far, fov=45)
            mvp = projection @ view
            with self.mesh_program:
                self.mesh_program['mvp'] = mvp
                self.mesh_program['albedo_texture'] = 0
                self.mesh_program['influence_buffer'] = 1
                self.mesh_program['bone_matrices'] = 2
                self.palette_buffer.bind(2)
                self.mesh_program['light_direction'] = self._light_direction()
                if self.wireframe:
                    gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
                for index, (vertex_list, texture, flip_v, influences) in enumerate(self.mesh_lists):
                    self.mesh_program['has_albedo'] = texture is not None
                    self.mesh_program['flip_v'] = flip_v
                    self.mesh_program['base_color'] = ((1.0, 1.0, 1.0) if texture is not None
                                                       else PALETTE[index % len(PALETTE)])
                    influences.bind(1)
                    gl.glActiveTexture(gl.GL_TEXTURE0)
                    if texture is not None:
                        texture.bind()
                    vertex_list.draw(gl.GL_TRIANGLES)
                gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
                for unit in (1, 2):
                    gl.glActiveTexture(gl.GL_TEXTURE0 + unit)
                    gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
                gl.glActiveTexture(gl.GL_TEXTURE0)
                gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
            with self.line_program:
                self.line_program['mvp'] = mvp
                if self.show_grid and self.grid_list is not None:
                    self.line_program['line_color'] = (0.22, 0.27, 0.32)
                    self.grid_list.draw(gl.GL_LINES)
        framebuffer_width, framebuffer_height = self.get_framebuffer_size()
        gl.glViewport(0, 0, framebuffer_width, framebuffer_height)
        gl.glDisable(gl.GL_DEPTH_TEST)
        self.ui_batch.draw()
        self.invalid = False

    def on_resize(self, width, height):
        if hasattr(self, 'ui_batch'):
            self._layout()

    def on_expose(self):
        self.invalid = True

    def on_scale(self, scale, dpi):
        self.invalid = True

    def open_file(self):
        try:
            directory = self.scene.path.parent if self.scene else Path.cwd()
            path = choose_file(getattr(self, '_hwnd', None), directory)
            if path is not None:
                self.queue_load(path)
        except Exception as exc:
            self._status(str(exc), error=True)
        finally:
            self.invalid = True

    def on_file_drop(self, x, y, paths):
        if paths:
            self.queue_load(next((p for p in paths if Path(p).suffix.lower() == '.ds2md'), paths[0]))

    def on_mouse_press(self, x, y, button, modifiers):
        if button != mouse.LEFT:
            return
        self._ui_capture = False
        if 12 <= x <= 100 and self.height - 44 <= y <= self.height - 14:
            self._ui_capture = True
            self.open_file()
            return
        if y <= 170:
            self._ui_capture = True
            if self.timeline.x - 5 <= x <= self.timeline.x + self.timeline.width + 5 and 82 <= y <= 106:
                self._timeline_drag = True
                self._resume_after_seek = bool(self.animator and self.animator.playing)
                if self.animator is not None:
                    self.animator.playing = False
                self._sync_clock()
                self._seek_timeline(x)
                return
            for action, (rectangle, _) in self.animation_buttons.items():
                if (rectangle.x <= x <= rectangle.x + rectangle.width and
                        rectangle.y <= y <= rectangle.y + rectangle.height):
                    if action == 'previous':
                        self.cycle_clip(-1)
                    elif action == 'next':
                        self.cycle_clip(1)
                    elif action == 'play':
                        self.toggle_playback()
                    elif action == 'bind':
                        self.select_clip(None)
                    elif action == 'repeat':
                        self.toggle_repeat()
                    elif self.animator is not None:
                        self.set_animation_speed(self.animator.speed *
                                                 (0.5 if action == 'slower' else 2.0))
                    return

    def on_mouse_release(self, x, y, button, modifiers):
        if button == mouse.LEFT:
            if self._timeline_drag:
                self._seek_timeline(x)
                self._timeline_drag = False
                if self.animator is not None:
                    self.animator.playing = self._resume_after_seek
                self._sync_clock()
                self._update_animation_ui()
                self.invalid = True
            self._ui_capture = False

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if self._timeline_drag:
            self._seek_timeline(x)
            return
        if self._ui_capture:
            return
        if self.scene is None:
            return
        if buttons & (mouse.RIGHT | mouse.MIDDLE):
            eye, _ = self._camera()
            forward = (self.target - eye).normalize()
            right = forward.cross(Vec3(0, 0, 1)).normalize()
            up = right.cross(forward).normalize()
            _, height = self._model_viewport()
            units = 2 * self.distance * math.tan(math.radians(22.5)) / height
            self.target += (right * -dx + up * -dy) * units
        elif buttons & mouse.LEFT:
            if modifiers & key.MOD_SHIFT:
                self.light_yaw -= dx * 0.008
                self.light_pitch = max(math.radians(-89), min(math.radians(89),
                                                             self.light_pitch - dy * 0.008))
            else:
                self.yaw -= dx * 0.008
                self.pitch = max(math.radians(-89), min(math.radians(89), self.pitch - dy * 0.008))
        self.invalid = True

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        if self.scene is not None:
            factor = math.exp(max(-20, min(20, -scroll_y * 0.12)))
            self.distance = max(self.radius * 0.02, min(self.radius * 1000, self.distance * factor))
            self.invalid = True

    def on_key_press(self, symbol, modifiers):
        if symbol == key.O and modifiers & key.MOD_CTRL:
            self.open_file()
        elif symbol == key.SPACE:
            self.toggle_playback()
        elif symbol == key.BRACKETLEFT:
            self.cycle_clip(-1)
        elif symbol == key.BRACKETRIGHT:
            self.cycle_clip(1)
        elif symbol in (key.LEFT, key.RIGHT, key.UP, key.DOWN):
            if self.animator is not None:
                step = 1.0 if symbol in (key.UP, key.DOWN) or modifiers & key.MOD_SHIFT else 0.1
                direction = -1 if symbol in (key.LEFT, key.DOWN) else 1
                self.seek_animation(self.animator.time_seconds + direction * step)
        elif symbol == key.HOME:
            self.seek_animation(0)
        elif symbol in (key.EQUAL, key.PLUS, key.NUM_ADD):
            if self.animator is not None:
                self.set_animation_speed(self.animator.speed * 2)
        elif symbol in (key.MINUS, key.NUM_SUBTRACT):
            if self.animator is not None:
                self.set_animation_speed(self.animator.speed * 0.5)
        elif symbol == key.R:
            self.toggle_repeat()
        elif symbol == key.B:
            self.select_clip(None)
        elif symbol == key.F:
            self.frame_model()
        elif symbol == key.L:
            self.reset_light()
        elif symbol == key.W:
            self.wireframe = not self.wireframe
            self._update_controls()
        elif symbol == key.G:
            self.show_grid = not self.show_grid
            self._update_controls()
        elif symbol == key.ESCAPE:
            self.close()

    def close(self):
        pyglet.clock.unschedule(self._load_pending)
        pyglet.clock.unschedule(self.update_animation)
        if self.context is not None and hasattr(self, 'mesh_lists'):
            self.switch_to()
            self._release_scene_resources()
            self.animator = None
            self.scene = None
            self.mesh_program.delete()
            self.line_program.delete()
        super().close()


def main():
    parser = argparse.ArgumentParser(description='Local DS2MD albedo and animation viewer.')
    parser.add_argument('model', nargs='?', type=Path, help='Optional .ds2md file to open')
    args = parser.parse_args()
    try:
        ModelViewer(args.model)
        # The clock calls draw; its invalid guard avoids idle GPU redraws.
        pyglet.app.run()
    except Exception as exc:
        print(f'DS2MD viewer could not start: {exc}\nOpenGL 3.3 and pyglet 2.1.19 are required.', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
