import argparse
import json
import os
import sys
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml

from load_data import generate_miss_loader
from utils import weight_matrix


def _sanitize_for_json(payload: Any) -> Any:
    """Recursively convert non-serialisable values into JSON-friendly types."""

    if isinstance(payload, dict):
        return {key: _sanitize_for_json(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [_sanitize_for_json(item) for item in payload]
    if isinstance(payload, (float, np.floating)):
        value = float(payload)
        if np.isfinite(value):
            return value
        return None
    if isinstance(payload, (int, np.integer)):
        return int(payload)
    return payload


def _safe_label(label: str) -> str:
    return label.replace(" ", "_").replace("/", "-")


def _prepare_run_data(
    data_settings: Dict[str, Any],
    learnable_flag: bool,
    device: torch.device,
    log_strategy_steps: bool,
    seed: int,
) -> Dict[str, Any]:
    from main import seed_torch  # imported lazily to avoid argparse conflicts

    seed_torch(seed)
    train_loader, valid_loader, test_loader, mean, std, A = generate_miss_loader(
        data_settings["dataset"],
        data_settings["miss_mechanism"],
        data_settings["miss_ratio"],
        data_settings["seqlen"],
        data_settings["batch_size"],
        data_settings["val_batch_size"],
        data_settings["test_batch_size"],
    )

    adj_mx = weight_matrix(A)

    base_model_kwargs = dict(
        device=device,
        num_nodes=A.shape[0],
        seqlen=data_settings["seqlen"],
        in_channels=data_settings["in_channel"],
        hidden_dim=data_settings["hidden_size"],
        st_block=data_settings["st_block"],
        K=data_settings["K"],
        d_model=data_settings["d_model"],
        n_heads=data_settings["n_heads"],
        adj_mx=adj_mx,
        learnable=learnable_flag,
        log_strategy_steps=log_strategy_steps,
    )

    return {
        "train_loader": train_loader,
        "valid_loader": valid_loader,
        "test_loader": test_loader,
        "mean": mean,
        "std": std,
        "adjacency": A,
        "base_model_kwargs": base_model_kwargs,
        "dataset": data_settings["dataset"],
        "miss_mechanism": data_settings["miss_mechanism"],
        "miss_ratio": data_settings["miss_ratio"],
    }


def _summarise_run(
    framework,
    diagnostics: Dict[str, Any],
    label: str,
    weight_strategy: str,
    missing_token_mode: str,
    seed: int,
) -> Dict[str, Any]:
    metrics = diagnostics.get("ensemble_metrics", {})
    weights = diagnostics.get("ensemble_weights", [])
    variant_metrics = diagnostics.get("variant_metrics", {})
    summary_variant_metrics = {
        name: {metric: float(value) for metric, value in metrics_dict.items()}
        for name, metrics_dict in variant_metrics.items()
    }
    variant_layout_serialised = [list(item) for item in getattr(framework, "variant_layout", [])]

    return {
        "label": label,
        "mae": float(metrics.get("mae", float("nan"))),
        "rmse": float(metrics.get("rmse", float("nan"))),
        "mape": float(metrics.get("mape", float("nan"))),
        "weights": [float(w) for w in weights],
        "variant_metrics": summary_variant_metrics,
        "variant_layout": variant_layout_serialised,
        "best_epoch": int(getattr(framework, "best_epoch", -1)),
        "best_ensemble_rmse": float(getattr(framework, "best_ensemble_score", float("nan"))),
        "weight_strategy": weight_strategy,
        "missing_token_mode": missing_token_mode,
        "seed": seed,
    }


def _train_and_evaluate(
    label: str,
    data_bundle: Dict[str, Any],
    device: torch.device,
    training_cfg: Dict[str, Any],
    save_root: str,
    result_root: str,
    missing_token_mode: str,
    variant_layout: Optional[Sequence[Tuple[str, str, str]]] = None,
    weight_override: Optional[Iterable[float]] = None,
    seed: int = 0,
    return_framework: bool = False,
):
    from main import UDCGLEnsemble, predict

    base_kwargs = dict(data_bundle["base_model_kwargs"])
    base_kwargs["missing_token_mode"] = missing_token_mode

    framework = UDCGLEnsemble(
        device=device,
        base_model_kwargs=base_kwargs,
        lr=training_cfg["lr"],
        delta=training_cfg["delta"],
        l1_lambda=training_cfg["l1_lambda"],
        l2_lambda=training_cfg["l2_lambda"],
        patience_limit=training_cfg["patience"],
        log_strategy_steps=training_cfg["log_strategy_steps"],
        variant_layout=variant_layout,
    )

    framework.train(
        data_bundle["train_loader"],
        data_bundle["valid_loader"],
        training_cfg["epochs"],
    )

    if framework.best_epoch != -1:
        print(
            f"[{label}] Best ensemble epoch: {framework.best_epoch} "
            f"(ensemble RMSE {framework.best_ensemble_score:.6f})"
        )

    label_safe = _safe_label(label)
    save_dir = os.path.join(save_root, "ablation", label_safe)
    framework.save_best_models(
        save_dir,
        data_bundle["dataset"],
        data_bundle["miss_mechanism"],
        data_bundle["miss_ratio"],
        seed,
    )

    result_dir = os.path.join(result_root, "ablation", label_safe)
    diagnostics = predict(
        framework,
        result_dir,
        data_bundle["test_loader"],
        data_bundle["mean"],
        data_bundle["std"],
        data_bundle["dataset"],
        data_bundle["miss_ratio"],
        data_bundle["miss_mechanism"],
        seed,
        collect_outputs=True,
        weight_override=weight_override,
    )
    if diagnostics is None:
        raise RuntimeError(f"{label} evaluation did not return diagnostics payload.")

    summary = _summarise_run(
        framework,
        diagnostics,
        label,
        weight_strategy="adaptive" if weight_override is None else "override",
        missing_token_mode=missing_token_mode,
        seed=seed,
    )

    payload = {
        "summary": summary,
        "diagnostics": diagnostics,
        "framework": framework if return_framework else None,
    }
    return payload



def _print_summary(title: str, summary: Dict[str, Any]) -> None:
    mae = summary.get("mae", float("nan"))
    rmse = summary.get("rmse", float("nan"))
    mape = summary.get("mape", float("nan"))
    weights = summary.get("weights", [])
    print(
        f"  {title}: MAE = {mae:.6f}, RMSE = {rmse:.6f}, MAPE = {mape * 100:.6f}%"
    )
    if weights:
        formatted = ", ".join(f"{w:.4f}" for w in weights)
        print(f"    Weights: [{formatted}]")



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run cailiCO ablation studies for UDCGL and report final metrics.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/cailiCO.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument(
        "--learnable",
        type=int,
        default=1,
        help="1 enables learnable positional embeddings (default pipeline).",
    )
    parser.add_argument(
        "--miss_ratio",
        type=float,
        default=None,
        help="Override missing ratio from the configuration file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="analysis/ablation",
        help="Directory to store ablation result summaries.",
    )
    args = parser.parse_args()

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0]]
        from main import seed_torch  # noqa: F401  (ensures dependency availability)
    finally:
        sys.argv = original_argv

    with open(args.config_path, "r", encoding="utf-8") as fb:
        config = yaml.load(fb, Loader=yaml.FullLoader)

    dataset = config["data"]["dataset"]
    miss_mechanism = config["data"]["miss_mechanism"]
    miss_ratio = float(config["data"]["miss_ratio"])
    if args.miss_ratio is not None:
        miss_ratio = args.miss_ratio

    batch_size = int(config["data"]["batch_size"])
    val_batch_size = int(config["data"]["val_batch_size"])
    test_batch_size = int(config["data"]["test_batch_size"])
    seqlen = int(config["model"]["seqlen"])
    st_block = int(config["model"]["st_block"])
    in_channel = int(config["model"]["in_channel"])
    hidden_size = int(config["model"]["hidden_size"])
    K = int(config["model"]["K"])
    d_model = int(config["model"]["d_model"])
    n_heads = int(config["model"]["n_heads"])

    epochs = int(config["train"]["epochs"])
    lr = float(config["train"]["lr"])
    save_path = config["train"]["save_model_path"]
    result_path = config["train"]["result_path"]
    patience_limit = int(config["train"].get("patience", 20))
    log_strategy_steps = bool(config["train"].get("log_strategy_steps", True))

    ensemble_cfg = config["train"].get("ensemble", {})
    delta = float(ensemble_cfg.get("delta", 0.5))
    lambda_default = float(ensemble_cfg.get("lambda", 1e-4))
    l1_lambda = float(ensemble_cfg.get("lambda_l1", lambda_default))
    l2_lambda = float(ensemble_cfg.get("lambda_l2", lambda_default))

    device = torch.device(
        f"cuda:{int(config['train']['cuda'])}" if torch.cuda.is_available() else "cpu"
    )

    learnable_flag = bool(args.learnable)
    torch.set_num_threads(10)

    data_settings = {
        "dataset": dataset,
        "miss_mechanism": miss_mechanism,
        "miss_ratio": miss_ratio,
        "seqlen": seqlen,
        "batch_size": batch_size,
        "val_batch_size": val_batch_size,
        "test_batch_size": test_batch_size,
        "st_block": st_block,
        "in_channel": in_channel,
        "hidden_size": hidden_size,
        "K": K,
        "d_model": d_model,
        "n_heads": n_heads,
    }

    training_cfg = {
        "epochs": epochs,
        "lr": lr,
        "delta": delta,
        "l1_lambda": l1_lambda,
        "l2_lambda": l2_lambda,
        "patience": patience_limit,
        "log_strategy_steps": log_strategy_steps,
    }

    os.makedirs(args.output_dir, exist_ok=True)

    results: Dict[str, Any] = {
        "metadata": {
            "dataset": dataset,
            "miss_mechanism": miss_mechanism,
            "miss_ratio": miss_ratio,
            "config_path": args.config_path,
            "seed": args.seed,
        }
    }

    print("=== Experiment 1: Learnable Uncertainty Embedding ===")
    baseline_data = _prepare_run_data(
        data_settings,
        learnable_flag=learnable_flag,
        device=device,
        log_strategy_steps=log_strategy_steps,
        seed=args.seed,
    )
    baseline_payload = _train_and_evaluate(
        "Full UDCGL",
        baseline_data,
        device,
        training_cfg,
        save_path,
        result_path,
        missing_token_mode="learnable",
        variant_layout=None,
        seed=args.seed,
        return_framework=True,
    )
    baseline_summary = baseline_payload["summary"]
    _print_summary("Full UDCGL", baseline_summary)

    zero_data = _prepare_run_data(
        data_settings,
        learnable_flag=learnable_flag,
        device=device,
        log_strategy_steps=log_strategy_steps,
        seed=args.seed,
    )
    zero_payload = _train_and_evaluate(
        "UDCGL-ZeroFill",
        zero_data,
        device,
        training_cfg,
        save_path,
        result_path,
        missing_token_mode="zero",
        variant_layout=None,
        seed=args.seed,
        return_framework=False,
    )
    zero_summary = zero_payload["summary"]
    _print_summary("UDCGL-ZeroFill", zero_summary)

    results["experiment_1"] = {
        "full_udcgl": {
            "summary": baseline_summary,
            "diagnostics": _sanitize_for_json(baseline_payload["diagnostics"]),
        },
        "udcgl_zero_fill": {
            "summary": zero_summary,
            "diagnostics": _sanitize_for_json(zero_payload["diagnostics"]),
        },
    }

    print("\n=== Experiment 2: Heterogeneous Objective Ensemble ===")
    single_head_data = _prepare_run_data(
        data_settings,
        learnable_flag=learnable_flag,
        device=device,
        log_strategy_steps=log_strategy_steps,
        seed=args.seed,
    )
    single_head_layout = [("UDCGL-SingleHead", "l1", "l2")]
    single_head_payload = _train_and_evaluate(
        "UDCGL-SingleHead",
        single_head_data,
        device,
        training_cfg,
        save_path,
        result_path,
        missing_token_mode="learnable",
        variant_layout=single_head_layout,
        seed=args.seed,
        return_framework=False,
    )
    single_head_summary = single_head_payload["summary"]
    _print_summary("UDCGL-SingleHead", single_head_summary)

    results["experiment_2"] = {
        "full_udcgl": {
            "summary": baseline_summary,
            "diagnostics": _sanitize_for_json(baseline_payload["diagnostics"]),
        },
        "udcgl_single_head": {
            "summary": single_head_summary,
            "diagnostics": _sanitize_for_json(single_head_payload["diagnostics"]),
        },
    }

    print("\n=== Experiment 3: Dynamic Weighting Strategy ===")
    baseline_framework = baseline_payload["framework"]
    if baseline_framework is None:
        raise RuntimeError("Baseline framework missing for dynamic weighting evaluation.")
    avg_weights = np.ones(len(baseline_framework.variants), dtype=np.float32)
    from main import predict  # local import ensures updated signature

    avg_label = "UDCGL-Avg"
    avg_label_safe = _safe_label(avg_label)
    avg_result_dir = os.path.join(result_path, "ablation", avg_label_safe)
    avg_diagnostics = predict(
        baseline_framework,
        avg_result_dir,
        baseline_data["test_loader"],
        baseline_data["mean"],
        baseline_data["std"],
        baseline_data["dataset"],
        baseline_data["miss_ratio"],
        baseline_data["miss_mechanism"],
        args.seed,
        collect_outputs=True,
        weight_override=avg_weights,
    )
    if avg_diagnostics is None:
        raise RuntimeError("Static averaging evaluation did not return diagnostics payload.")
    avg_summary = _summarise_run(
        baseline_framework,
        avg_diagnostics,
        avg_label,
        weight_strategy="static_average",
        missing_token_mode="learnable",
        seed=args.seed,
    )
    _print_summary("UDCGL-Avg", avg_summary)

    results["experiment_3"] = {
        "full_udcgl": {
            "summary": baseline_summary,
            "diagnostics": _sanitize_for_json(baseline_payload["diagnostics"]),
        },
        "udcgl_avg": {
            "summary": avg_summary,
            "diagnostics": _sanitize_for_json(avg_diagnostics),
        },
    }

    output_path = os.path.join(
        args.output_dir, f"{dataset}_ablation_results_seed{args.seed}.json"
    )
    with open(output_path, "w", encoding="utf-8") as fb:
        json.dump(_sanitize_for_json(results), fb, indent=2, ensure_ascii=False)
    print(f"\n[Ablation] Saved aggregated results to: {output_path}")


if __name__ == "__main__":
    main()
