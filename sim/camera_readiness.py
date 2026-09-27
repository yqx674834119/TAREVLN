"""Bounded camera warm-up without advancing physics or inventing observations."""


def render_rgba(world, camera, minimum_renders=4, maximum_renders=40):
    for count in range(1, maximum_renders+1):
        world.render()
        if count < minimum_renders:
            continue
        rgba=camera.get_rgba()
        if rgba is not None and rgba.ndim == 3 and rgba.shape[2] >= 3 and rgba.size:
            return rgba, count
    raise RuntimeError(f'Camera produced no valid frame after {maximum_renders} renders')
