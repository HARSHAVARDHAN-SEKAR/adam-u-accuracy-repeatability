#!/usr/bin/env python3
"""
make_comparison_video.py -- side-by-side video of the same reach in both control
modes, for the presentation.

    python3 scripts/make_comparison_video.py                     # gravity: PD vs PD + feed-forward
    python3 scripts/make_comparison_video.py \
        --left pd_gff:nominal --right pd_gff:perturbed           # payload and friction
    MUJOCO_GL=egl python3 scripts/make_comparison_video.py       # headless machines

Each panel is one (control mode, plant condition) pair, written as mode:condition.
Both panels show the commanded point as a blue sphere and the live position error,
so the difference between them is visible without any commentary.

Writes results/comparison_<left>_vs_<right>_<target>.mp4 when ffmpeg is available,
otherwise an animated GIF of the same frames.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from src.experiment import REPO_ROOT, Setup  # noqa: E402
from src.trial import run_trial  # noqa: E402

W, H = 640, 480   # MuJoCo's default offscreen framebuffer; larger needs a <global offheight> in the model
EVERY = 40                  # simulation steps between captured frames
FPS = 25
MODE_LABEL = {"pd": "Vendor example as shipped", "pd_gff": "With gravity feed-forward"}
MODE_LAW = {"pd": "tau = kp (q_cmd - q) - kd dq", "pd_gff": "tau = g(q_cmd) + kp (q_cmd - q) - kd dq"}
COND_LABEL = {"nominal": "ideal plant", "perturbed": "payload, friction, offsets, noise"}
BAND_TOP, BAND_BOTTOM = 96, 52


def _font(size, bold=False):
    """Matplotlib ships DejaVu, so a real font is available without new deps."""
    try:
        import matplotlib
        name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
        return ImageFont.truetype(os.path.join(os.path.dirname(matplotlib.__file__),
                                               "mpl-data", "fonts", "ttf", name), size)
    except Exception:
        return ImageFont.load_default()


F_TITLE, F_SUB, F_PANEL, F_ERR, F_LAW = (_font(26, True), _font(17), _font(20, True),
                                         _font(22, True), _font(15))


def capture(setup, mode, target, condition, approach, seed):
    """Run one trial, returning a frame every EVERY steps plus the live error."""
    plant, data = setup.plant_for(condition)
    renderer = mujoco.Renderer(plant.model, H, W)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = target.pos + np.array([-0.05, 0.0, -0.05])
    cam.distance, cam.azimuth, cam.elevation = 1.25, 140.0, -8.0

    frames, errors = [], []
    state = {"n": 0}

    def on_step():
        state["n"] += 1
        if state["n"] % EVERY:
            return
        mujoco.mj_kinematics(plant.model, data)
        renderer.update_scene(data, cam)
        geom = renderer.scene.geoms[renderer.scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.015, 0, 0]),
                            target.pos, np.eye(3).flatten(), np.array([0.1, 0.45, 1.0, 1.0], np.float32))
        renderer.scene.ngeom += 1
        frames.append(Image.fromarray(renderer.render()))
        errors.append(float(np.linalg.norm(data.site_xpos[plant.tcp_site_id] - target.pos) * 1000.0))

    rng = np.random.default_rng(seed) if condition.is_stochastic else None
    res = run_trial(plant, data, setup.controller(mode), condition, setup.start_config(target, approach),
                    target.q_cmd, setup.cfg["protocol"], rng, sim_cfg=setup.cfg["simulation"], on_step=on_step)
    renderer.close()
    return frames, errors, res


def describe(mode, cond_name, vary_mode, vary_cond):
    """Panel heading: name whichever of the two factors actually differs."""
    if vary_mode and vary_cond:
        return f"{MODE_LABEL[mode]} - {COND_LABEL.get(cond_name, cond_name)}"
    if vary_cond:
        return COND_LABEL.get(cond_name, cond_name).capitalize()
    return MODE_LABEL[mode]


def compose(left, right, err_l, err_r, headings, laws, subtitle, target_name, t):
    """One side-by-side frame with titles and the live error."""
    canvas = Image.new("RGB", (2 * W, H + BAND_TOP + BAND_BOTTOM), (252, 252, 251))
    canvas.paste(left, (0, BAND_TOP))
    canvas.paste(right, (W, BAND_TOP))
    d = ImageDraw.Draw(canvas)
    d.text((18, 10), f"ADAM-U right hand reaching target {target_name}", font=F_TITLE, fill=(11, 11, 11))
    d.text((18, 42), f"t = {t:4.2f} s      blue sphere = commanded point      {subtitle}",
           font=F_SUB, fill=(82, 81, 78))
    for i, (heading, law, err) in enumerate(zip(headings, laws, (err_l, err_r))):
        x = i * W + 18
        d.text((x, 68), heading, font=F_PANEL, fill=(11, 11, 11))
        colour = (216, 90, 48) if err > 5.0 else (29, 158, 117)
        d.text((x, H + BAND_TOP + 10), f"error {err:9.3f} mm", font=F_ERR, fill=colour)
        d.text((x, H + BAND_TOP + 34), law, font=F_LAW, fill=(130, 129, 124))
    d.line([(W, BAND_TOP), (W, H + BAND_TOP)], fill=(214, 212, 204), width=2)
    return canvas


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--target", default="P1")
    ap.add_argument("--approach", default="plus")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--left", default="pd:nominal", help="mode:condition for the left panel")
    ap.add_argument("--right", default="pd_gff:nominal", help="mode:condition for the right panel")
    args = ap.parse_args()

    setup = Setup(model_path=args.model_path)
    target = next(t for t in setup.targets if t.name == args.target)
    panels = []
    for spec in (args.left, args.right):
        mode, _, cond = spec.partition(":")
        panels.append((mode, cond or "nominal"))
    vary_mode = panels[0][0] != panels[1][0]
    vary_cond = panels[0][1] != panels[1][1]
    if not (vary_mode or vary_cond):
        ap.error("--left and --right are the same run; nothing to compare")

    runs = []
    for mode, cond_name in panels:
        print(f"rendering {mode} / {cond_name} ...", flush=True)
        frames, errors, res = capture(setup, mode, target, setup.condition(cond_name),
                                      args.approach, args.seed)
        print(f"  {len(frames)} frames, final error {np.linalg.norm(res.tcp_pos - target.pos) * 1000:.3f} mm")
        runs.append((frames, errors))

    headings = [describe(m, c, vary_mode, vary_cond) for m, c in panels]
    laws = [MODE_LAW[m] for m, _ in panels]
    subtitle = ("same command and gains in both panels" if vary_mode
                else f"same controller ({MODE_LABEL[panels[0][0]].lower()}) in both panels")
    n = min(len(runs[0][0]), len(runs[1][0]))
    dt = setup.nominal.model.opt.timestep * EVERY
    composed = [compose(runs[0][0][i], runs[1][0][i], runs[0][1][i], runs[1][1][i],
                        headings, laws, subtitle, target.name, (i + 1) * dt) for i in range(n)]

    out_dir = os.path.join(REPO_ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)
    tag = lambda m, c: f"{m}-{c}"  # noqa: E731
    stem = os.path.join(out_dir, f"comparison_{tag(*panels[0])}_vs_{tag(*panels[1])}_{target.name}")

    if shutil.which("ffmpeg"):
        tmp = tempfile.mkdtemp()
        for i, im in enumerate(composed):
            im.save(os.path.join(tmp, f"{i:05d}.png"))
        out = stem + ".mp4"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", str(FPS),
                        "-i", os.path.join(tmp, "%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", out], check=True)
        shutil.rmtree(tmp)
    else:
        out = stem + ".gif"
        composed[0].save(out, save_all=True, append_images=composed[1:],
                         duration=int(1000 / FPS), loop=0, optimize=True)
        print("ffmpeg not found, wrote a GIF instead")
    print("wrote", os.path.relpath(out, REPO_ROOT), f"({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
