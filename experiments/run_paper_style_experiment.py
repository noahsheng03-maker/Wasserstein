from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.deepc.data_structures import partition_data_matrix
from src.deepc.dr_deepc import WassersteinDRDeePC
from src.deepc.radius_schedule import AdaptiveRadiusConfig, AdaptiveRadiusScheduler
from src.deepc.systems import LinearHoverSystem
from src.utils.config import load_yaml


def sample_input_sequence(config: dict, system: LinearHoverSystem, horizon: int, rng: np.random.Generator) -> np.ndarray:
    objective = config["objective"]
    constraints = config["constraints"]
    u_std = config["offline_data"]["input_std"]
    sequence = np.vstack(
        [
            objective["hover_thrust"] + u_std * rng.standard_normal(horizon),
            u_std * rng.standard_normal(horizon),
            u_std * rng.standard_normal(horizon),
        ]
    )
    lower = np.asarray(constraints["input_lower"], dtype=float).reshape(-1, 1)
    upper = np.asarray(constraints["input_upper"], dtype=float).reshape(-1, 1)
    return np.clip(sequence, lower, upper)


def build_offline_batches(config: dict, matrix_type: str | None = None, n_batches: int | None = None, horizon: int | None = None):
    rng = np.random.default_rng(config["seed"])
    system = LinearHoverSystem(
        dt=config["system"]["dt"],
        process_noise_std=config["system"]["process_noise_std"],
        measurement_noise_std=config["system"]["measurement_noise_std"],
        hover_thrust=config["objective"]["hover_thrust"],
    )
    t_ini = config["controller"]["T_ini"]
    t_f = config["controller"]["T_f"]
    horizon = config["offline_data"]["horizon"] if horizon is None else horizon
    n_batches = config["offline_data"]["N"] if n_batches is None else n_batches
    matrix_type = config["controller"]["matrix_type"] if matrix_type is None else matrix_type
    repeated_input = sample_input_sequence(config, system, horizon, rng)
    up = uf = None
    yp_batches = []
    yf_batches = []
    raw_batches = []
    for batch_idx in range(n_batches):
        if config["offline_data"].get("repeated_input", True):
            input_sequence = repeated_input
        else:
            input_sequence = sample_input_sequence(config, system, horizon, rng)
        x0 = np.zeros(system.state_dim)
        _, y_seq = system.rollout(input_sequence, x0=x0, rng=np.random.default_rng(config["seed"] + batch_idx + 1))
        u_p, u_f, y_p, y_f = partition_data_matrix(input_sequence, y_seq, t_ini, t_f, matrix_type)
        up = u_p
        uf = u_f
        yp_batches.append(y_p)
        yf_batches.append(y_f)
        raw_batches.append((input_sequence.copy(), y_seq.copy()))
    return system, up, uf, yp_batches, yf_batches, raw_batches


def build_controller(config: dict, up, uf, yp_batches, yf_batches):
    constraints = config["constraints"]
    objective = config["objective"]
    t_f = config["controller"]["T_f"]
    input_dim = len(constraints["input_lower"])
    output_dim = len(constraints["output_lower"])

    input_lower = np.tile(np.asarray(constraints["input_lower"], dtype=float), t_f)
    input_upper = np.tile(np.asarray(constraints["input_upper"], dtype=float), t_f)
    output_lower = np.tile(np.asarray(constraints["output_lower"], dtype=float), t_f)
    output_upper = np.tile(np.asarray(constraints["output_upper"], dtype=float), t_f)
    y_ref = np.tile(np.asarray(objective["y_ref"], dtype=float), t_f)

    input_weights = np.tile(
        np.asarray([objective["thrust_weight"], objective["rate_weight"], objective["rate_weight"]], dtype=float),
        t_f,
    )

    return WassersteinDRDeePC(
        up=up,
        uf=uf,
        yp_batches=yp_batches,
        yf_batches=yf_batches,
        input_lower=input_lower,
        input_upper=input_upper,
        output_lower=output_lower,
        output_upper=output_upper,
        y_ref=y_ref,
        alpha=config["controller"]["alpha"],
        l_obj=objective["l_obj"],
        l_con=objective["l_con"],
        input_weights=input_weights,
        output_weight=objective["output_weight"],
        past_weight=objective["past_weight"],
    )


def evaluate_closed_loop(config: dict, mode: str = "fixed", fixed_epsilon: float | None = None):
    system, up, uf, yp_batches, yf_batches, raw_batches = build_offline_batches(config)
    controller = build_controller(config, up, uf, yp_batches, yf_batches)
    t_ini = config["controller"]["T_ini"]
    horizon_sim = config["controller"]["horizon_sim"]
    control_horizon = config["controller"]["control_horizon"]

    u_hist = []
    y_hist = []
    x = np.zeros(system.state_dim)
    epsilon = config["controller"]["epsilon_nominal"] if fixed_epsilon is None else fixed_epsilon
    scheduler = None
    if mode == "adaptive":
        scheduler = AdaptiveRadiusScheduler(AdaptiveRadiusConfig(**config["adaptive_radius"]))
        epsilon = scheduler.epsilon

    rng = np.random.default_rng(config["seed"] + 1000)
    hover_thrust = float(config["objective"]["hover_thrust"])
    warm_u = np.vstack(
        [
            np.full(t_ini, hover_thrust),
            np.zeros(t_ini),
            np.zeros(t_ini),
        ]
    )
    _, warm_y = system.rollout(warm_u, x0=x, rng=np.random.default_rng(config["seed"] + 999))
    for k in range(t_ini):
        u_hist.append(warm_u[:, k].copy())
        y_hist.append(warm_y[:, k].copy())

    epsilon_trace = [epsilon]
    prediction_error_trace = []
    stress_trace = []
    shift_trace = []

    out_lower = np.asarray(config["constraints"]["output_lower"], dtype=float)
    out_upper = np.asarray(config["constraints"]["output_upper"], dtype=float)

    for t in range(horizon_sim):
        u_ini = np.asarray(u_hist[-t_ini:]).T.reshape(-1, order="F")
        y_ini = np.asarray(y_hist[-t_ini:]).T.reshape(-1, order="F")
        result = controller.solve(u_ini=u_ini, y_ini=y_ini, epsilon=epsilon)
        u_star = result.u_future.reshape(system.input_dim, -1, order="F")
        u_apply = u_star[:, 0]
        xs, ys = system.rollout(u_apply.reshape(system.input_dim, 1), x0=x, rng=rng)
        x = xs[:, -1]
        y_next = ys[:, -1]
        y_pred_first = result.y_future_samples[0].reshape(system.output_dim, -1, order="F")[:, 0]

        prediction_error = float(np.linalg.norm(y_next - y_pred_first))
        violation = np.maximum(y_next - out_upper, 0.0) + np.maximum(out_lower - y_next, 0.0)
        stress = float(np.max(violation))
        shift = prediction_error

        prediction_error_trace.append(prediction_error)
        stress_trace.append(stress)
        shift_trace.append(shift)

        u_hist.append(u_apply.copy())
        y_hist.append(y_next.copy())

        if scheduler is not None and (t + 1) % control_horizon == 0:
            epsilon = scheduler.update(prediction_error, stress, shift)
        epsilon_trace.append(epsilon)

    y_array = np.asarray(y_hist[t_ini:])
    y_ref = np.asarray(config["objective"]["y_ref"], dtype=float)
    tracking_error = float(np.sum((y_array - y_ref) ** 2))
    violation_rate = float(np.mean(np.any((y_array > out_upper) | (y_array < out_lower), axis=1)))
    violation_magnitudes = np.maximum(np.maximum(y_array - out_upper, out_lower - y_array), 0.0)
    tail_score = float(np.mean(np.sort(np.max(violation_magnitudes, axis=1))[-max(1, len(y_array)//10):]))
    return {
        "mode": mode,
        "epsilon_initial": epsilon_trace[0],
        "epsilon_final": epsilon_trace[-1],
        "tracking_error": tracking_error,
        "violation_rate": violation_rate,
        "tail_violation_score": tail_score,
        "epsilon_trace": epsilon_trace,
        "prediction_error_trace": prediction_error_trace,
        "stress_trace": stress_trace,
        "shift_trace": shift_trace,
        "offline_columns": int(up.shape[1]),
        "offline_horizon": int(config["offline_data"]["horizon"]),
        "offline_batches": int(config["offline_data"]["N"]),
        "matrix_type": config["controller"]["matrix_type"],
        "t_ini": int(config["controller"]["T_ini"]),
        "t_f": int(config["controller"]["T_f"]),
        "raw_batches_shapes": [(u.shape, y.shape) for u, y in raw_batches],
    }


def clone_config(config: dict) -> dict:
    return copy.deepcopy(config)


def save_partial_frame(out_dir: Path, file_name: str, records: list[dict]) -> None:
    out_dir.mkdir(exist_ok=True)
    pd.DataFrame(records).to_csv(out_dir / file_name, index=False)


def run_nominal_comparison(config: dict, out_dir: Path | None = None) -> tuple[pd.DataFrame, dict]:
    records = []
    study_start = time.time()
    print("[nominal] running adaptive controller")
    adaptive = evaluate_closed_loop(config, mode="adaptive")
    if out_dir is not None:
        with (out_dir / "adaptive_trace.json").open("w", encoding="utf-8") as handle:
            json.dump(adaptive, handle, indent=2)
    for epsilon in config["evaluation"]["epsilon_grid"]:
        epsilon_start = time.time()
        print(f"[nominal] running fixed epsilon={epsilon}")
        res = evaluate_closed_loop(config, mode="fixed", fixed_epsilon=epsilon)
        records.append(
            {
                "study": "nominal_comparison",
                "controller_mode": f"fixed_{epsilon}",
                "epsilon_setting": epsilon,
                "tracking_error": res["tracking_error"],
                "violation_rate": res["violation_rate"],
                "tail_violation_score": res["tail_violation_score"],
                "epsilon_final": res["epsilon_final"],
                "offline_columns": res["offline_columns"],
                "offline_batches": res["offline_batches"],
                "offline_horizon": res["offline_horizon"],
                "matrix_type": res["matrix_type"],
                "t_ini": res["t_ini"],
            }
        )
        print(f"[nominal] finished epsilon={epsilon} in {time.time() - epsilon_start:.1f}s")
        if out_dir is not None:
            save_partial_frame(out_dir, "paper_style_nominal_comparison.csv", records)
    records.append(
        {
            "study": "nominal_comparison",
            "controller_mode": "adaptive",
            "epsilon_setting": adaptive["epsilon_initial"],
            "tracking_error": adaptive["tracking_error"],
            "violation_rate": adaptive["violation_rate"],
            "tail_violation_score": adaptive["tail_violation_score"],
            "epsilon_final": adaptive["epsilon_final"],
            "offline_columns": adaptive["offline_columns"],
            "offline_batches": adaptive["offline_batches"],
            "offline_horizon": adaptive["offline_horizon"],
            "matrix_type": adaptive["matrix_type"],
            "t_ini": adaptive["t_ini"],
        }
    )
    if out_dir is not None:
        save_partial_frame(out_dir, "paper_style_nominal_comparison.csv", records)
    print(f"[nominal] complete in {time.time() - study_start:.1f}s")
    return pd.DataFrame(records), adaptive


def run_epsilon_sweep(config: dict, out_dir: Path | None = None) -> pd.DataFrame:
    records = []
    study_start = time.time()
    for n_batches in config["evaluation"]["N_grid"]:
        local_config = clone_config(config)
        local_config["offline_data"]["N"] = int(n_batches)
        for epsilon in config["evaluation"]["epsilon_grid"]:
            case_start = time.time()
            print(f"[epsilon] running N={n_batches}, epsilon={epsilon}")
            res = evaluate_closed_loop(local_config, mode="fixed", fixed_epsilon=epsilon)
            records.append(
                {
                    "study": "epsilon_sweep",
                    "N": int(n_batches),
                    "epsilon": epsilon,
                    "tracking_error": res["tracking_error"],
                    "violation_rate": res["violation_rate"],
                    "tail_violation_score": res["tail_violation_score"],
                    "offline_columns": res["offline_columns"],
                    "matrix_type": res["matrix_type"],
                    "t_ini": res["t_ini"],
                }
            )
            print(f"[epsilon] finished N={n_batches}, epsilon={epsilon} in {time.time() - case_start:.1f}s")
            if out_dir is not None:
                save_partial_frame(out_dir, "paper_style_epsilon_sweep.csv", records)
    print(f"[epsilon] complete in {time.time() - study_start:.1f}s")
    return pd.DataFrame(records)


def run_horizon_and_matrix_sweep(config: dict, out_dir: Path | None = None) -> pd.DataFrame:
    records = []
    study_start = time.time()
    for matrix_type in config["evaluation"]["matrix_types"]:
        for horizon in config["evaluation"]["T_grid"]:
            local_config = clone_config(config)
            local_config["controller"]["matrix_type"] = matrix_type
            local_config["offline_data"]["horizon"] = int(horizon)
            case_start = time.time()
            print(f"[horizon-matrix] running matrix_type={matrix_type}, horizon={horizon}")
            res = evaluate_closed_loop(local_config, mode="fixed", fixed_epsilon=config["controller"]["epsilon_nominal"])
            records.append(
                {
                    "study": "horizon_matrix_sweep",
                    "matrix_type": matrix_type,
                    "offline_horizon": int(horizon),
                    "offline_columns": res["offline_columns"],
                    "tracking_error": res["tracking_error"],
                    "violation_rate": res["violation_rate"],
                    "tail_violation_score": res["tail_violation_score"],
                    "t_ini": res["t_ini"],
                }
            )
            print(
                f"[horizon-matrix] finished matrix_type={matrix_type}, horizon={horizon} in {time.time() - case_start:.1f}s"
            )
            if out_dir is not None:
                save_partial_frame(out_dir, "paper_style_horizon_matrix_sweep.csv", records)
    print(f"[horizon-matrix] complete in {time.time() - study_start:.1f}s")
    return pd.DataFrame(records)


def run_tini_sweep(config: dict, out_dir: Path | None = None) -> pd.DataFrame:
    records = []
    study_start = time.time()
    for t_ini in config["evaluation"]["T_ini_grid"]:
        local_config = clone_config(config)
        local_config["controller"]["T_ini"] = int(t_ini)
        case_start = time.time()
        print(f"[tini] running T_ini={t_ini}")
        res = evaluate_closed_loop(local_config, mode="fixed", fixed_epsilon=config["controller"]["epsilon_nominal"])
        records.append(
            {
                "study": "tini_sweep",
                "t_ini": int(t_ini),
                "tracking_error": res["tracking_error"],
                "violation_rate": res["violation_rate"],
                "tail_violation_score": res["tail_violation_score"],
                "offline_columns": res["offline_columns"],
                "matrix_type": res["matrix_type"],
            }
        )
        print(f"[tini] finished T_ini={t_ini} in {time.time() - case_start:.1f}s")
        if out_dir is not None:
            save_partial_frame(out_dir, "paper_style_tini_sweep.csv", records)
    print(f"[tini] complete in {time.time() - study_start:.1f}s")
    return pd.DataFrame(records)


def write_outputs(out_dir: Path, frames: dict[str, pd.DataFrame], adaptive_trace: dict) -> None:
    out_dir.mkdir(exist_ok=True)
    for name, frame in frames.items():
        frame.to_csv(out_dir / f"{name}.csv", index=False)
    with (out_dir / "adaptive_trace.json").open("w", encoding="utf-8") as handle:
        json.dump(adaptive_trace, handle, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paper-style DR-DeePC experiment studies.")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "experiments" / "paper_base.yaml"),
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--study",
        choices=["all", "nominal", "epsilon", "horizon-matrix", "tini"],
        default="all",
        help="Select which paper-style study to run.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_yaml(Path(args.config))
    out_dir = REPO_ROOT / "outputs"

    frames: dict[str, pd.DataFrame] = {}
    adaptive_trace: dict = {}

    if args.study in {"all", "nominal"}:
        nominal, adaptive_trace = run_nominal_comparison(config, out_dir=out_dir)
        frames["paper_style_nominal_comparison"] = nominal
    if args.study in {"all", "epsilon"}:
        frames["paper_style_epsilon_sweep"] = run_epsilon_sweep(config, out_dir=out_dir)
    if args.study in {"all", "horizon-matrix"}:
        frames["paper_style_horizon_matrix_sweep"] = run_horizon_and_matrix_sweep(config, out_dir=out_dir)
    if args.study in {"all", "tini"}:
        frames["paper_style_tini_sweep"] = run_tini_sweep(config, out_dir=out_dir)

    if not adaptive_trace:
        adaptive_trace = evaluate_closed_loop(config, mode="adaptive")

    write_outputs(out_dir, frames, adaptive_trace)

    for name, frame in frames.items():
        print(f"\n=== {name} ===")
        print(frame)


if __name__ == "__main__":
    main()
