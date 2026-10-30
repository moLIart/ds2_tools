"""Local, static DS2MD albedo viewer. Run: python view_ds2md.py [MODEL.ds2md]."""

from __future__ import annotations

import argparse
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


MESH_VERTEX = """#version 330 core
in vec3 position;
in vec3 normal;
in vec2 uv_coords;
uniform mat4 mvp;
out vec3 surface_normal;
out vec2 texture_uv;
void main() {
    gl_Position = mvp * vec4(position, 1.0);
    surface_normal = normal;
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
        self.stats_label = pyglet.text.Label('Static geometry | Albedo 0 meshes / 0 textures | No animation',
                                            font_size=10, x=14, anchor_y='top', batch=self.ui_batch)
        self.note_label = pyglet.text.Label('', font_size=10, x=14, anchor_y='top',
                                           multiline=True, width=1100, batch=self.ui_batch,
                                           color=(180, 190, 202, 255))
        self.controls_label = pyglet.text.Label('', font_size=10, x=14, y=42, batch=self.ui_batch)
        self.status_label = pyglet.text.Label('Open or drop a DS2MD model to begin.', font_size=10,
                                             x=14, y=28, anchor_y='top', multiline=True,
                                             width=1100, batch=self.ui_batch)
        self.empty_label = pyglet.text.Label('Drop a .ds2md file here\nor click Open (Ctrl+O)',
                                            font_size=18, anchor_x='center', anchor_y='center',
                                            align='center', multiline=True, width=520,
                                            batch=self.ui_batch, color=(175, 187, 201, 255))
        self._layout()
        self._update_controls()
        gl.glClearColor(0.075, 0.09, 0.115, 1.0)
        gl.glDisable(gl.GL_CULL_FACE)
        if initial_path is not None:
            self.queue_load(initial_path)

    def _layout(self):
        self.open_button.y = self.height - 44
        self.open_label.y = self.height - 29
        self.title_label.y = self.height - 18
        self.title_label.width = max(100, self.width - 130)
        self.stats_label.y = self.height - 56
        self.note_label.y = self.height - 76
        self.note_label.width = max(100, self.width - 28)
        self.status_label.width = max(100, self.width - 28)
        self.empty_label.x = self.width / 2
        self.empty_label.y = self.height / 2
        self.invalid = True

    def _update_controls(self):
        self.controls_label.text = (f'LMB orbit | Shift+LMB light | RMB/MMB pan | Wheel zoom | '
                                    f'F frame | L reset light | W wire [{"on" if self.wireframe else "off"}] | '
                                    f'G grid [{"on" if self.show_grid else "off"}] | Ctrl+O open')
        self.invalid = True

    def _status(self, text, error=False):
        self.status_label.text = text
        self.status_label.color = (255, 155, 145, 255) if error else (190, 205, 220, 255)
        self.invalid = True

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
        try:
            scene = load_scene(path)
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
                vertex_list = self.mesh_program.vertex_list_indexed(
                    len(mesh.positions) // 3, gl.GL_TRIANGLES, mesh.indices,
                    position=('f', mesh.positions), normal=('f', mesh.normals),
                    uv_coords=('f', mesh.uv_coords))
                allocated.append(vertex_list)
                meshes.append((vertex_list, texture,
                               mesh.albedo_path is not None and
                               mesh.albedo_path.suffix.casefold() == '.tga'))
            low, high = scene.bounds_min, scene.bounds_max
            center = Vec3(*((a + b) * 0.5 for a, b in zip(low, high)))
            radius = max(math.sqrt(sum((b - a) ** 2 for a, b in zip(low, high))) * 0.5, 0.001)
            extent = radius * 1.5
            z = low[2] - radius * 0.005
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
            for texture in textures.values():
                if texture is not None:
                    texture.delete()
            self._status(f'Cannot open {path.name}: {exc}', error=True)
            return

        for vertex_list, _, _ in self.mesh_lists:
            vertex_list.delete()
        for texture in self.albedo_textures.values():
            if texture is not None:
                texture.delete()
        if self.grid_list is not None:
            self.grid_list.delete()
        self.scene = scene
        self.mesh_lists = meshes
        self.albedo_textures = textures
        self.grid_list = grid
        self.radius = radius
        self.frame_model()
        self.set_caption(f'{scene.path.name} - DS2MD Geometry Viewer')
        self.title_label.text = f'{scene.path.name}  |  {scene.name}'
        vertices = sum(len(mesh.positions) // 3 for mesh in scene.meshes)
        triangles = sum(len(mesh.indices) // 3 for mesh in scene.meshes)
        textured_meshes = sum(texture is not None for _, texture, _ in meshes)
        texture_count = sum(texture is not None for texture in textures.values())
        self.stats_label.text = (f'DS2MD {scene.version} | {len(scene.meshes):,} meshes | '
                                 f'{vertices:,} vertices | {triangles:,} triangles | '
                                 f'Albedo {textured_meshes}/{len(meshes)} meshes / '
                                 f'{texture_count} textures | No animation')
        self.note_label.text = scene.note
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

    def frame_model(self):
        if self.scene is not None:
            self.target = Vec3(*((a + b) * 0.5 for a, b in
                                  zip(self.scene.bounds_min, self.scene.bounds_max)))
        self.yaw = math.radians(-55)
        self.pitch = math.radians(20)
        half_fov = math.radians(45) / 2
        aspect = max(self.width, 1) / max(self.height, 1)
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
            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glDepthFunc(gl.GL_LEQUAL)
            gl.glDisable(gl.GL_CULL_FACE)
            eye, view = self._camera()
            center = Vec3(*((a + b) * 0.5 for a, b in
                            zip(self.scene.bounds_min, self.scene.bounds_max)))
            far = max((eye - center).length() + self.radius * 5, self.radius * 10)
            near = max(self.radius * 0.0001, self.distance * 0.001)
            projection = Mat4.perspective_projection(max(self.width, 1) / max(self.height, 1),
                                                    near, far, fov=45)
            mvp = projection @ view
            with self.mesh_program:
                self.mesh_program['mvp'] = mvp
                self.mesh_program['albedo_texture'] = 0
                self.mesh_program['light_direction'] = self._light_direction()
                if self.wireframe:
                    gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
                for index, (vertex_list, texture, flip_v) in enumerate(self.mesh_lists):
                    self.mesh_program['has_albedo'] = texture is not None
                    self.mesh_program['flip_v'] = flip_v
                    self.mesh_program['base_color'] = ((1.0, 1.0, 1.0) if texture is not None
                                                       else PALETTE[index % len(PALETTE)])
                    if texture is not None:
                        texture.bind()
                    vertex_list.draw(gl.GL_TRIANGLES)
                gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
                gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
            with self.line_program:
                self.line_program['mvp'] = mvp
                if self.show_grid and self.grid_list is not None:
                    self.line_program['line_color'] = (0.22, 0.27, 0.32)
                    self.grid_list.draw(gl.GL_LINES)
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
        if (button == mouse.LEFT and 12 <= x <= 100
                and self.height - 44 <= y <= self.height - 14):
            self.open_file()

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if self.scene is None:
            return
        if buttons & (mouse.RIGHT | mouse.MIDDLE):
            eye, _ = self._camera()
            forward = (self.target - eye).normalize()
            right = forward.cross(Vec3(0, 0, 1)).normalize()
            up = right.cross(forward).normalize()
            units = 2 * self.distance * math.tan(math.radians(22.5)) / max(self.height, 1)
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
        if self.context is not None and hasattr(self, 'mesh_lists'):
            self.switch_to()
            for vertex_list, _, _ in self.mesh_lists:
                vertex_list.delete()
            self.mesh_lists.clear()
            if self.grid_list is not None:
                self.grid_list.delete()
                self.grid_list = None
            for texture in self.albedo_textures.values():
                if texture is not None:
                    texture.delete()
            self.albedo_textures.clear()
        super().close()


def main():
    parser = argparse.ArgumentParser(description='Local DS2MD static albedo viewer (no animation).')
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
