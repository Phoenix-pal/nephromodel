"""Standalone animation comparing two legacy dynamic trajectories.

This replaces a notebook-export script that relied on IPython display,
hard-coded frame indices, and loaded the same file twice.  It is deliberately
separate from the pytest suite in ``tests/test_model.py``.

Run from the project root:

    python src/kidney_model/test.py

or save a custom comparison:

    python src/kidney_model/test.py --left-file dynamic_stable_v2.npy \
        --right-file conv_to_third_ss.npy --output results/bistability.gif
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_profiles(path: str | Path) -> dict[str, np.ndarray]:
    """Load the documented legacy ``8*N + 7`` trajectory layout."""
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    data = np.load(path, mmap_mode="r")
    if data.ndim != 2 or (data.shape[1] - 7) % 8 != 0:
        raise ValueError(f"{path} is not a legacy trajectory with 8*N + 7 columns")
    n = (data.shape[1] - 7) // 8
    faces = n + 1
    return {
        "name": path.stem,
        "x_face": np.linspace(0.0, 1.0, faces),
        "x_cell": np.linspace(1.0 / (2 * n), 1.0 - 1.0 / (2 * n), n),
        "D": 2.0 * data[:, 3 * faces:4 * faces],
        "A": 2.0 * data[:, 4 * faces:5 * faces],
        "C": data[:, 5 * faces:6 * faces],
        "I": 2.0 * data[:, 6 * faces:7 * n + 6] + data[:, 7 * n + 6:8 * n + 6],
    }


def build_animation(left: dict[str, np.ndarray], right: dict[str, np.ndarray], frames: int):
    """Return a matplotlib animation with time aligned by trajectory fraction."""
    if frames < 2:
        raise ValueError("frames must be at least 2")
    left_indices = np.linspace(0, len(left["D"]) - 1, frames, dtype=int)
    right_indices = np.linspace(0, len(right["D"]) - 1, frames, dtype=int)
    ymax = max(
        float(np.max(left[key])) for key in ("D", "A", "C", "I")
    )
    ymax = max(ymax, max(float(np.max(right[key])) for key in ("D", "A", "C", "I")))

    figure, axis = plt.subplots(figsize=(9, 5), dpi=140)
    axis.set(
        xlim=(0, 1),
        ylim=(0, ymax * 1.05),
        xlabel="medullary position: cortex (0) to papilla (1)",
        ylabel="legacy osmolarity (a.u.)",
    )
    axis.grid(True)

    colors = {"D": "red", "A": "green", "C": "blue", "I": "magenta"}
    labels = {
        "D": "D descending limb: cortex -> papilla (downward flow)",
        "A": "A ascending limb: papilla -> cortex (upward flow)",
        "C": "C collecting duct: cortex -> papilla (downward flow)",
        "I": "I interstitium: no tubular axial flow direction",
    }
    left_lines, right_lines = {}, {}
    for key in ("D", "A", "C", "I"):
        x = left["x_cell"] if key == "I" else left["x_face"]
        left_lines[key], = axis.plot(
            x, left[key][0], "--", color=colors[key],
            label=f"{left['name']} (dashed): {labels[key]}",
        )
        x = right["x_cell"] if key == "I" else right["x_face"]
        right_lines[key], = axis.plot(
            x, right[key][0], "-", color=colors[key],
            label=f"{right['name']} (solid): {labels[key]}",
        )
    axis.legend(ncol=1, fontsize=6, loc="upper left")

    def update(frame: int):
        li, ri = left_indices[frame], right_indices[frame]
        for key in ("D", "A", "C", "I"):
            left_lines[key].set_ydata(left[key][li])
            right_lines[key].set_ydata(right[key][ri])
        axis.set_title(
            f"trajectory fraction: {frame / (frames - 1):.1%}  |  "
            f"{left['name']} row={li}, {right['name']} row={ri}"
        )
        return [*left_lines.values(), *right_lines.values()]

    return animation.FuncAnimation(figure, update, frames=frames, interval=50, blit=False), figure


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Compare two legacy kidney-model trajectories as a GIF")
    parser.add_argument("--left-file", default="dynamic_stable_v2.npy")
    parser.add_argument("--right-file", default="conv_to_third_ss.npy")
    parser.add_argument("--output", default="results/bistability.gif")
    parser.add_argument("--frames", type=int, default=161)
    parser.add_argument("--show", action="store_true", help="open an interactive window after saving")
    args = parser.parse_args(argv)

    left = load_profiles(args.left_file)
    right = load_profiles(args.right_file)
    ani, figure = build_animation(left, right, args.frames)
    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    ani.save(output, writer=animation.PillowWriter(fps=20))
    print(f"saved animation to {output}")
    if args.show:
        plt.show()
    else:
        plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
