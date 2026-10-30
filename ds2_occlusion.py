"""World-space mesh grouping and conservative occlusion-query eligibility."""

from dataclasses import dataclass
import math
import operator


@dataclass(frozen=True, slots=True)
class OcclusionGroup:
    mesh_indices: tuple[int, ...]
    bounds_min: tuple
    bounds_max: tuple


def build_occlusion_groups(meshes, group_size=64) -> tuple[OcclusionGroup, ...]:
    """Split mesh centroids spatially; enclose complete serialized world AABBs.

    Mesh transforms are deliberately ignored: serialized bounds already include
    them. Padding covers float32 rounding when the boxes reach the GPU, including
    boxes with zero extent.
    """
    group_size = operator.index(group_size)
    if group_size < 1:
        raise ValueError('group_size must be positive')
    bounds = [(tuple(mesh['bounds_min']), tuple(mesh['bounds_max'])) for mesh in meshes]
    centroids = [tuple(low[a] * .5 + high[a] * .5 for a in range(3))
                 for low, high in bounds]
    groups = []

    def partition(indices):
        if len(indices) <= group_size:
            low = tuple(min(bounds[i][0][a] for i in indices) for a in range(3))
            high = tuple(max(bounds[i][1][a] for i in indices) for a in range(3))
            padding = tuple(1e-6 * max(1., abs(low[a]), abs(high[a]), high[a]-low[a])
                            for a in range(3))
            groups.append(OcclusionGroup(
                tuple(indices),
                tuple(low[a]-padding[a] for a in range(3)),
                tuple(high[a]+padding[a] for a in range(3))))
            return
        spans = [max(centroids[i][a] for i in indices)
                 - min(centroids[i][a] for i in indices) for a in range(3)]
        axis = max(range(3), key=lambda a: spans[a])
        indices.sort(key=lambda i: (centroids[i][axis], i))
        middle = len(indices) // 2
        partition(indices[:middle])
        partition(indices[middle:])

    if bounds:
        partition(list(range(len(bounds))))
    return tuple(groups)


def group_query_safe(group, eye, forward, near) -> bool:
    """Return whether the entire padded box is safely past the near plane.

    Forward is normalized here so near remains a world-space distance. Boxes
    containing or touching the eye, or touching/crossing the near plane, must
    render unconditionally rather than risk a clipped query proxy.
    """
    low, high = group.bounds_min, group.bounds_max
    if all(low[a] <= eye[a] <= high[a] for a in range(3)):
        return False
    length = math.hypot(*forward)
    if not math.isfinite(length) or length == 0 or not math.isfinite(near):
        return False
    direction = tuple(component / length for component in forward)
    minimum_depth = sum(((low[a] if direction[a] >= 0 else high[a])-eye[a])
                        * direction[a] for a in range(3))
    scale = max(1., abs(near), *(abs(v) for v in eye),
                *(abs(v) for v in low), *(abs(v) for v in high))
    return minimum_depth > near + 1e-6 * scale
