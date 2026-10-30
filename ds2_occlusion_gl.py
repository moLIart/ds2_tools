"""OpenGL 3.3 bounding-box queries for the DS2 map viewer.

Import after configuring pyglet's GL options, and create/delete with the owning
GL context current. Query results are ready-only diagnostics, not CPU culling
inputs; rendering can use the current batch's query IDs with GL_QUERY_NO_WAIT.
"""

import ctypes
from contextlib import ExitStack

from pyglet import gl
from pyglet.graphics.shader import Shader, ShaderProgram


_VERTEX = """#version 330 core
layout(location=0) in vec3 position;
uniform mat4 mvp;
uniform vec3 bounds_min;
uniform vec3 bounds_max;
void main() {
    gl_Position = mvp * vec4(mix(bounds_min, bounds_max, position), 1.0);
}
"""
_FRAGMENT = """#version 330 core
void main() {
}
"""

# Eight shared corners and twelve triangles close all six faces. These are
# unit-box data only: no game geometry is copied or transformed a second time.
_CORNERS = (
    0., 0., 0.,  1., 0., 0.,  1., 1., 0.,  0., 1., 0.,
    0., 0., 1.,  1., 0., 1.,  1., 1., 1.,  0., 1., 1.,
)
_INDICES = (
    0, 2, 1, 0, 3, 2,
    4, 5, 6, 4, 6, 7,
    0, 1, 5, 0, 5, 4,
    3, 7, 6, 3, 6, 2,
    0, 4, 7, 0, 7, 3,
    1, 2, 6, 1, 6, 5,
)


class OcclusionQueries:
    """Own one shared box and a group-indexed ctypes array of query IDs.

    The caller must exclude eye-containing and near-plane-intersecting boxes,
    and issue against the current frame's accumulated opaque depth. After
    issue(), invalidate render-state caches and explicitly rebind the main
    shader. The helper leaves color writes enabled, depth comparison GL_LESS,
    no program bound and VAO zero; other modified state is caller-owned.
    """

    def __init__(self, groups):
        self.groups = tuple(groups)
        self.queries = (gl.GLuint * len(self.groups))()
        self.program = None
        self._vao = gl.GLuint()
        self._buffers = (gl.GLuint * 2)()
        self._issued = bytearray(len(self.groups))
        self._available = gl.GLuint()
        self._result = gl.GLuint()
        self._deleted = False
        try:
            # Successfully constructed Pyglet shaders/programs are explicitly
            # owned, including when a later construction step fails.
            with ExitStack() as shaders:
                vertex = Shader(_VERTEX, 'vertex')
                shaders.callback(vertex.delete)
                fragment = Shader(_FRAGMENT, 'fragment')
                shaders.callback(fragment.delete)
                self.program = ShaderProgram(vertex, fragment)
            self._bounds_min = gl.glGetUniformLocation(self.program.id, b'bounds_min')
            self._bounds_max = gl.glGetUniformLocation(self.program.id, b'bounds_max')

            vertices = (gl.GLfloat * len(_CORNERS))(*_CORNERS)
            indices = (gl.GLubyte * len(_INDICES))(*_INDICES)
            gl.glGenVertexArrays(1, ctypes.byref(self._vao))
            gl.glGenBuffers(2, self._buffers)
            gl.glBindVertexArray(self._vao)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._buffers[0])
            gl.glBufferData(gl.GL_ARRAY_BUFFER, ctypes.sizeof(vertices), vertices,
                            gl.GL_STATIC_DRAW)
            gl.glEnableVertexAttribArray(0)
            gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0,
                                     ctypes.c_void_p(0))
            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self._buffers[1])
            gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, ctypes.sizeof(indices), indices,
                            gl.GL_STATIC_DRAW)
            if self.groups:
                gl.glGenQueries(len(self.groups), self.queries)
        except BaseException:
            # Pyglet's ShaderException is a BaseException, not an Exception.
            self.delete()
            raise
        finally:
            try:
                gl.glBindVertexArray(0)
            finally:
                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

    def issue(self, group_indices, mvp) -> None:
        """Issue a single depth-only query batch without retrieving any results.

        Wireframe must bypass querying. Bounds are already padded world AABBs;
        mvp is projection * view, without a mesh/model transform.
        """
        if self._deleted:
            raise RuntimeError('OcclusionQueries has been deleted')
        try:
            self.program.use()
            self.program['mvp'] = mvp
            gl.glBindVertexArray(self._vao)
            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glDepthFunc(gl.GL_LEQUAL)
            gl.glDepthMask(gl.GL_FALSE)
            gl.glColorMask(gl.GL_FALSE, gl.GL_FALSE, gl.GL_FALSE, gl.GL_FALSE)
            gl.glDisable(gl.GL_CULL_FACE)
            gl.glDisable(gl.GL_BLEND)
            gl.glDisable(gl.GL_POLYGON_OFFSET_FILL)
            gl.glDisable(gl.GL_POLYGON_OFFSET_LINE)
            gl.glDisable(gl.GL_POLYGON_OFFSET_POINT)
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
            for index in group_indices:
                self._issued[index] = False
                group = self.groups[index]
                gl.glUniform3f(self._bounds_min, *group.bounds_min)
                gl.glUniform3f(self._bounds_max, *group.bounds_max)
                gl.glBeginQuery(gl.GL_ANY_SAMPLES_PASSED, self.queries[index])
                try:
                    gl.glDrawElements(gl.GL_TRIANGLES, len(_INDICES), gl.GL_UNSIGNED_BYTE,
                                      ctypes.c_void_p(0))
                finally:
                    gl.glEndQuery(gl.GL_ANY_SAMPLES_PASSED)
                self._issued[index] = True
        finally:
            # Each restoration is attempted even if another GL call raises.
            try:
                gl.glColorMask(gl.GL_TRUE, gl.GL_TRUE, gl.GL_TRUE, gl.GL_TRUE)
            finally:
                try:
                    gl.glDepthFunc(gl.GL_LESS)
                finally:
                    try:
                        ShaderProgram.stop()
                    finally:
                        gl.glBindVertexArray(0)

    def results(self, group_indices) -> dict[int, bool]:
        """Return only completed requested results, for optional display.

        A query not issued by this owner is skipped. Availability is always
        checked before reading a result; no waiting or previous-result cache
        is used. Callers should request only indices issued for their frame.
        """
        if self._deleted:
            raise RuntimeError('OcclusionQueries has been deleted')
        ready = {}
        for index in group_indices:
            if not self._issued[index]:
                continue
            gl.glGetQueryObjectuiv(self.queries[index], gl.GL_QUERY_RESULT_AVAILABLE,
                                  ctypes.byref(self._available))
            if self._available.value:
                gl.glGetQueryObjectuiv(self.queries[index], gl.GL_QUERY_RESULT,
                                      ctypes.byref(self._result))
                ready[index] = bool(self._result.value)
        return ready

    def delete(self) -> None:
        """Release every owned resource once, including partial construction.

        No finalizer makes GL calls: the owner must delete before destroying
        its context. All releases are attempted even if one of them raises.
        """
        if self._deleted:
            return
        self._deleted = True
        with ExitStack() as resources:
            if self.program is not None:
                resources.callback(self.program.delete)
                self.program = None
            if any(self._buffers):
                resources.callback(gl.glDeleteBuffers, 2, self._buffers)
                self._buffers = (gl.GLuint * 2)()
            if self._vao.value:
                resources.callback(gl.glDeleteVertexArrays, 1, ctypes.byref(self._vao))
                self._vao = gl.GLuint()
            if any(self.queries):
                resources.callback(gl.glDeleteQueries, len(self.queries), self.queries)
                self.queries = (gl.GLuint * len(self.groups))()
