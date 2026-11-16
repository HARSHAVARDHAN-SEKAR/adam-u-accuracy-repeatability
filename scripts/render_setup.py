#!/usr/bin/env python3
"""
render_setup.py -- optional: draw docs/setup.png (robot at target P1, the five
targets as blue spheres).  Needs offscreen OpenGL; if it fails, try
    MUJOCO_GL=egl python3 scripts/render_setup.py      (or MUJOCO_GL=osmesa)
Nothing else in the repository depends on this picture.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from src.experiment import REPO_ROOT, Setup  # noqa: E402
from src.model_builder import ARM_SLICE  # noqa: E402


def main():
    s = Setup(model_path=sys.argv[1] if len(sys.argv) > 1 else None)
    model = s.nominal.model
    data = mujoco.MjData(model)
    q19 = np.zeros(19)
    q19[ARM_SLICE] = s.targets[0].q_cmd
    data.qpos[s.nominal.qadr] = q19
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, 480, 640)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.25, -0.2, 1.35]
    cam.distance, cam.azimuth, cam.elevation = 1.6, 140.0, -12.0
    renderer.update_scene(data, cam)
    for t in s.targets:
        geom = renderer.scene.geoms[renderer.scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.012, 0, 0]), t.pos,
                            np.eye(3).flatten(), np.array([0.1, 0.4, 1.0, 1.0], dtype=np.float32))
        renderer.scene.ngeom += 1
    out = os.path.join(REPO_ROOT, "docs", "setup.png")
    plt.imsave(out, renderer.render())
    print("wrote", os.path.relpath(out, REPO_ROOT))


if __name__ == "__main__":
    main()
