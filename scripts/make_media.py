"""Generate the README media: a side-by-side tracking GIF and a results chart.

  docs/media/tracking_delay.gif      baseline vs residual policy, same seed,
                                     figure-eight with 100 ms control delay
  docs/media/results_light.png       baseline vs policy RMSE per condition
  docs/media/results_dark.png        (same chart for GitHub dark mode)

Usage:
    python scripts/make_media.py --model results/final/sac_residual_s0.zip
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from envs.tracking_env import PandaTrackingEnv  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "media")

# ── GIF ─────────────────────────────────────────────────────────────────────
W, H = 420, 380          # per-panel render size
STRIP = 70               # error-plot strip under each panel
BG = (26, 26, 25)
COL_TARGET = [0.15, 0.90, 0.25, 1.0]
COL_REF = [0.95, 0.80, 0.10, 0.6]
COL_EE = [0.30, 0.60, 1.00, 0.9]


def _font(size):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _add_sphere(scn, pos, radius, rgba):
    if scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([radius, 0, 0]), np.asarray(pos, float),
                        np.eye(3).flatten(), np.asarray(rgba, np.float32))
    scn.ngeom += 1


def rollout_frames(mode, policy, delay, traj, seed, t0, t1, every):
    env = PandaTrackingEnv(trajectory=traj, control_mode=mode,
                           delay_steps=delay, randomize=False, seed=seed)
    obs, _ = env.reset(seed=seed)
    renderer = mujoco.Renderer(env.model, H, W)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = env.traj.center
    cam.distance, cam.azimuth, cam.elevation = 0.62, 180.0, -4.0

    frames, errs, trail = [], [], []
    while env.t < t1:
        a = (np.zeros(7, np.float32) if policy is None
             else policy.predict(obs, deterministic=True)[0])
        obs, _, _, _, info = env.step(a)
        trail.append(info["ee_pos"].copy())
        if env.t < t0:
            continue
        errs.append(info["ee_error"])
        if env.step_count % every:
            continue
        renderer.update_scene(env.data, camera=cam)
        scn = renderer.scene
        for h in np.arange(0.0, env.traj.period, 0.08):  # one full period ahead
            _add_sphere(scn, env.ref_pos(env.t + h), 0.0035, COL_REF)
        for p in trail[-150::2]:                      # EE path, last 3 s
            _add_sphere(scn, p, 0.004, COL_EE)
        _add_sphere(scn, info["target_pos"], 0.012, COL_TARGET)
        frames.append((renderer.render().copy(), list(errs)))
    renderer.close()
    env.close()
    return frames


def compose(left, right, title, labels, err_max):
    font, small = _font(17), _font(14)
    panels = []
    for (img, errs), label in zip((left, right), labels):
        img = img.copy()
        img[(img.sum(axis=2) == 0)] = BG               # empty sky -> surface
        canvas = Image.new("RGB", (W, H + STRIP), BG)
        canvas.paste(Image.fromarray(img), (0, 0))
        d = ImageDraw.Draw(canvas)
        rmse = 1000 * np.sqrt(np.mean(np.square(errs)))
        d.text((12, 10), label, font=font, fill=(255, 255, 255))
        d.text((12, 34), f"running RMSE {rmse:4.1f} mm", font=small,
               fill=(195, 194, 183))
        # error trace
        y0, y1 = H + 8, H + STRIP - 10
        d.line([(10, y1), (W - 10, y1)], fill=(80, 80, 76))
        if len(errs) > 1:
            xs = np.linspace(10, W - 10, len(errs))
            ys = y1 - (y1 - y0) * np.clip(np.array(errs) / err_max, 0, 1)
            d.line(list(zip(xs, ys)), fill=(57, 135, 229), width=2)
        d.text((W - 118, y0 - 4), f"error (0–{1000 * err_max:.0f} mm)",
               font=_font(11), fill=(150, 149, 142))
        panels.append(canvas)
    out = Image.new("RGB", (2 * W + 6, H + STRIP + 34), BG)
    ImageDraw.Draw(out).text((12, 8), title, font=font, fill=(255, 255, 255))
    out.paste(panels[0], (0, 34))
    out.paste(panels[1], (W + 6, 34))
    return out


def make_gif(model_path, path, delay=5, traj="figure8", seed=3):
    from stable_baselines3 import SAC
    policy = SAC.load(model_path, device="cpu")
    kw = dict(delay=delay, traj=traj, seed=seed, t0=2.0, t1=11.0, every=3)
    base = rollout_frames("baseline", None, **kw)
    pol = rollout_frames("residual", policy, **kw)
    err_max = 0.02
    title = (f"Figure-eight, {20 * delay} ms control delay  ·  green: target  ·  "
             "yellow: path ahead  ·  blue: actual path")
    frames = [compose(b, p, title,
                      ("Classical diff-IK only", "Diff-IK + residual SAC"),
                      err_max)
              for b, p in zip(base, pol)]
    # Downscale and quantise so the GIF stays small enough for a README.
    frames = [f.resize((int(f.width * 0.72), int(f.height * 0.72)),
                       Image.LANCZOS).quantize(colors=64, method=Image.MEDIANCUT)
              for f in frames]
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=60, loop=0, optimize=True)
    rb = 1000 * np.sqrt(np.mean(np.square(base[-1][1])))
    rp = 1000 * np.sqrt(np.mean(np.square(pol[-1][1])))
    print(f"gif: {len(frames)} frames, baseline {rb:.1f} mm, policy {rp:.1f} mm "
          f"-> {path} ({os.path.getsize(path) / 1e6:.1f} MB)")


# ── results chart ───────────────────────────────────────────────────────────
THEMES = {
    "light": dict(surface="#fcfcfb", text="#0b0b0b", text2="#52514e",
                  grid="#e4e3df", base="#a3a29c", policy="#2a78d6"),
    "dark": dict(surface="#1a1a19", text="#ffffff", text2="#c3c2b7",
                 grid="#383835", base="#6b6a65", policy="#3987e5"),
}
CONDS = ["clean", "noise", "delay_60ms", "delay_100ms", "noise+delay"]
COND_LABELS = ["clean", "noise", "60 ms\ndelay", "100 ms\ndelay",
               "noise +\ndelay"]
TRAJS = [("figure8", "Figure-eight"), ("circle", "Circle"),
         ("random", "Random (aperiodic)")]


def load_benchmark(path):
    rows = {}
    pat = re.compile(r"\| (\w+) \| ([\w+]+) \| ([\d.]+) \| ([\d.]+) ± ([\d.]+)"
                     r" \| ([+-]\d+)%")
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = pat.match(line)
            if m:
                rows[(m[1], m[2])] = tuple(float(x) for x in m.groups()[2:])
    return rows


def make_chart(bench, path, theme):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = THEMES[theme]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "text.color": t["text"], "axes.labelcolor": t["text2"],
                         "xtick.color": t["text2"], "ytick.color": t["text2"]})
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True,
                             facecolor=t["surface"])
    x = np.arange(len(CONDS))
    bw = 0.36
    for ax, (key, name) in zip(axes, TRAJS):
        ax.set_facecolor(t["surface"])
        b = [bench[(key, c)][0] for c in CONDS]
        p = [bench[(key, c)][1] for c in CONDS]
        s = [bench[(key, c)][2] for c in CONDS]
        chg = [bench[(key, c)][3] for c in CONDS]  # as reported in benchmark.md
        ax.bar(x - bw / 2 - 0.01, b, bw, color=t["base"], label="Classical diff-IK")
        ax.bar(x + bw / 2 + 0.01, p, bw, color=t["policy"],
               label="Diff-IK + residual SAC")
        ax.errorbar(x + bw / 2 + 0.01, p, yerr=s, fmt="none",
                    ecolor=t["text2"], elinewidth=1, capsize=2)
        for xi, pi, si, ch in zip(x, p, s, chg):
            ax.text(xi + bw / 2 + 0.01, pi + si + 0.35, f"{ch:+.0f}%", ha="center",
                    fontsize=8.5, color=t["text"] if ch < 0 else t["text2"],
                    fontweight="bold" if ch < 0 else "normal")
        ax.set_title(name, color=t["text"], fontsize=11.5, loc="left", pad=8)
        ax.set_xticks(x, COND_LABELS, fontsize=8.5)
        ax.grid(axis="y", color=t["grid"], lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(t["grid"])
        ax.tick_params(length=0)
    axes[0].set_ylabel("tracking RMSE (mm)")
    axes[0].set_ylim(0, 12)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper right", ncol=2, frameon=False,
               bbox_to_anchor=(0.99, 1.0), fontsize=9.5)
    fig.suptitle("Tracking error by condition — lower is better", x=0.01,
                 ha="left", y=0.99, fontsize=13, color=t["text"])
    fig.text(0.01, 0.005, "Policy: mean ± std over 3 training seeds × 10 "
             "episodes. Labels: change vs baseline. Unreachable-path results "
             "in results/benchmark.md.", fontsize=8.5, color=t["text2"])
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(path, dpi=150, facecolor=t["surface"])
    plt.close(fig)
    print(f"chart ({theme}) -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="results/final/sac_residual_s0.zip")
    ap.add_argument("--benchmark", default="results/benchmark.md")
    ap.add_argument("--skip-gif", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    bench = load_benchmark(args.benchmark)
    for theme in THEMES:
        make_chart(bench, os.path.join(OUT, f"results_{theme}.png"), theme)
    if not args.skip_gif:
        make_gif(args.model, os.path.join(OUT, "tracking_delay.gif"))


if __name__ == "__main__":
    main()
