"""Command-line entry point for a small, reproducible model run."""

import argparse
from pathlib import Path

from .diagnostics import continuity_and_mass, state_summary
from .initialization import load_dynamic_state, make_initial_condition_from_file
from .parameters import build_parameters
from .solver import run_model_fsolve


def main(argv=None):
    parser = argparse.ArgumentParser(description="Simulate the dynamic renal concentrating mechanism")
    parser.add_argument("--N", type=int, default=30, help="spatial cells per compartment")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--dynamic-file", help="optional legacy .npy state file, such as conv_to_third_ss.npy")
    parser.add_argument("--row-index", type=int, default=-1, help="trajectory row; default is the final row")
    parser.add_argument("--impermeable-collecting-duct", action="store_true")
    parser.add_argument("--plot", action="store_true", help="show final-state plots")
    parser.add_argument("--save-plot-dir", help="save final-state plots to this directory")
    parser.add_argument("--animation", action="store_true", help="show the osmolarity animation")
    parser.add_argument("--save-animation", help="save animation as .html, .gif, or video")
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--gamma-label", type=float, default=1.3)
    args = parser.parse_args(argv)

    p = build_parameters(args.N, args.dt, not args.impermeable_collecting_duct)
    y_start = None
    dynamic_state = None
    if args.dynamic_file:
        dynamic_state = load_dynamic_state(args.dynamic_file, args.row_index)
        y_start = make_initial_condition_from_file(dynamic_state, p)
    result = run_model_fsolve(p, y_start=y_start, steps=args.steps, print_every=args.print_every)
    print("final state:", state_summary(result["final_state"], p))
    print("continuity:", continuity_and_mass(result["final_state"], p))

    if args.plot or args.save_plot_dir:
        from .visualization import plot_dynamic_source, plot_initial_condition, plot_result

        figures = []
        if dynamic_state is not None:
            figures.append(plot_dynamic_source(dynamic_state))
            figures.append(plot_initial_condition(result["history"][0], p))
        figures.extend(plot_result(result, p, state_index=-1, include_time=True))
        if args.save_plot_dir:
            output_dir = Path(args.save_plot_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            for index, figure in enumerate(figures, start=1):
                figure.savefig(output_dir / f"final_state_{index:02d}.png", dpi=150, bbox_inches="tight")
            print(f"saved {len(figures)} plots to {output_dir.resolve()}")
        if args.plot:
            import matplotlib.pyplot as plt

            plt.show()

    if args.animation or args.save_animation:
        from .visualization import animate_result, save_animation

        ani, figure = animate_result(
            result,
            p,
            frame_stride=args.frame_stride,
            gamma_label=args.gamma_label,
        )
        if args.save_animation:
            save_animation(ani, args.save_animation)
            print(f"saved animation to {Path(args.save_animation).resolve()}")
        if args.animation:
            import matplotlib.pyplot as plt

            plt.show()
        else:
            import matplotlib.pyplot as plt

            plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
