import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle as pk
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import argparse
import wandb
import time
import yaml

from utils import *
from models.model import UDCGLModel
from load_data import *

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config_path", type=str, default="configs/METR-LA.yaml", help="config filepath"
)
parser.add_argument("--seed", type=int, default=0, help="random seed")
parser.add_argument("--learnable", type=int, default=1, help="1: learnable; 0: fixed")
parser.add_argument("--miss_ratio", type=float, default=None, help="override missing ratio")
args = parser.parse_args()


def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_default_dtype(torch.float32)


class UDCGLEnsemble:
    """Manages four UDCGLModel variants and the adaptive ensemble pipeline."""

    VARIANT_LAYOUT = (
        ("UDCGL-A", "l1", "l1"),
        ("UDCGL-B", "l1", "l2"),
        ("UDCGL-C", "l2", "l1"),
        ("UDCGL-D", "l2", "l2"),
    )

    def __init__(
        self,
        device,
        base_model_kwargs,
        lr,
        delta=0.5,
        l1_lambda=1e-4,
        l2_lambda=1e-4,
        patience_limit=20,
        log_strategy_steps=True,
        variant_layout=None,
    ):
        self.device = device
        self.delta = float(delta)
        self.patience_limit = int(patience_limit)
        self.log_strategy_steps = log_strategy_steps
        self.eps = 1e-8
        self.variant_layout = (
            tuple(variant_layout)
            if variant_layout is not None
            else self.VARIANT_LAYOUT
        )
        if len(self.variant_layout) == 0:
            raise ValueError("variant_layout must contain at least one variant definition")
        self.variants = []
        self.accumulated_scores = torch.zeros(len(self.variant_layout), dtype=torch.float32)
        self.current_weights = torch.ones(len(self.variant_layout), dtype=torch.float32) / len(self.variant_layout)
        self.best_weights = self.current_weights.clone()
        self.best_states = None
        self.best_ensemble_score = float("inf")
        self.best_epoch = -1
        self.completed_epochs = 0
        self.patience = 0
        self.history = {
            "train_loss": [],
            "val_rmse": [],
            "weights": [],
        }

        base_kwargs = dict(base_model_kwargs)
        base_kwargs.setdefault("log_strategy_steps", log_strategy_steps)

        for idx, (name, recon_mode, reg_mode) in enumerate(self.variant_layout):
            variant_kwargs = dict(base_kwargs)
            if idx > 0:
                variant_kwargs["log_strategy_steps"] = False
            model = UDCGLModel(**variant_kwargs).to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            reg_lambda = float(l1_lambda) if reg_mode == "l1" else float(l2_lambda)
            self.variants.append(
                {
                    "name": name,
                    "model": model,
                    "optimizer": optimizer,
                    "recon_mode": recon_mode,
                    "reg_mode": reg_mode,
                    "reg_lambda": reg_lambda,
                    "last_train_loss": float("inf"),
                }
            )

        if self.log_strategy_steps:
            self._print_framework_overview(l1_lambda, l2_lambda)

    def _print_framework_overview(self, l1_lambda, l2_lambda):
        print("[UDCGLEnsemble] Step 4: Adaptive Validation-Guided Ensemble")
        variant_count = len(self.variant_layout)
        variant_phrase = "variants" if variant_count > 1 else "variant"
        print(
            f"  - {variant_count} base {variant_phrase} are instantiated with complementary reconstruction losses and regularizers:"
        )
        for name, recon_mode, reg_mode in self.variant_layout:
            lambda_val = l1_lambda if reg_mode == "l1" else l2_lambda
            print(
                f"    • {name}: {recon_mode.upper()} reconstruction + {reg_mode.upper()} regularization (λ = {lambda_val})"
            )
        if len(self.variant_layout) == 1:
            print("  - Single-variant configuration detected; ensemble weight degenerates to 1.0.")
        else:
            print(
                "  - Validation RMSE S^t(n) is tracked per variant, accumulated into A^t(n) = Σ_h S^t(h),"
            )
            print(
                "    and adaptive weights ε^t(n) = softmax(-δ · A^t(n)) with δ = {} are refreshed every epoch.".format(
                    self.delta
                )
            )
        print(
            "  - Ensemble predictions are late-fused as x̂ = Σ_t ε^t · x̂^t with gradients computed independently per variant."
        )

    def _compute_reconstruction_loss(self, preds, target, mask, mode):
        mask = mask.to(preds.dtype)
        denom = torch.clamp(mask.sum(), min=1.0)
        diff = preds - target
        if mode == "l1":
            loss = torch.sum(torch.abs(diff) * mask) / denom
        else:
            mse = torch.sum((diff * mask) ** 2) / denom
            loss = torch.sqrt(mse + self.eps)
        return loss

    def _compute_regularization(self, model, mode):
        reg_value = torch.tensor(0.0, device=self.device)
        for param in model.parameters():
            if mode == "l1":
                reg_value = reg_value + torch.sum(torch.abs(param))
            else:
                reg_value = reg_value + 0.5 * torch.sum(param ** 2)
        return reg_value

    def _snapshot_state(self, model):
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    def restore_best_states(self):
        if self.best_states is None:
            self.best_states = [self._snapshot_state(variant["model"]) for variant in self.variants]
        for variant, state in zip(self.variants, self.best_states):
            variant["model"].load_state_dict(state)
            variant["model"].to(self.device)
        self.current_weights = self.best_weights.clone()

    def _train_one_epoch(self, train_loader):
        for variant in self.variants:
            variant["model"].train()
            variant["epoch_losses"] = []

        for _, (x, m, y) in enumerate(train_loader):
            x = x.to(self.device)
            m = m.to(self.device)
            target = y[:, :, :1, :].to(self.device)
            observed_mask = m[:, :, :1, :].to(self.device)

            for variant in self.variants:
                variant["optimizer"].zero_grad()
                preds = variant["model"](x, m)
                recon_loss = self._compute_reconstruction_loss(
                    preds, target, observed_mask, variant["recon_mode"]
                )
                reg_term = self._compute_regularization(
                    variant["model"], variant["reg_mode"]
                )
                loss = recon_loss + variant["reg_lambda"] * reg_term
                loss.backward()
                variant["optimizer"].step()
                variant["epoch_losses"].append(loss.item())

        mean_losses = []
        for variant in self.variants:
            if variant["epoch_losses"]:
                variant["last_train_loss"] = float(np.mean(variant["epoch_losses"]))
            else:
                variant["last_train_loss"] = float("nan")
            mean_losses.append(variant["last_train_loss"])
        return mean_losses

    def _evaluate_validation(self, valid_loader):
        sq_errors = [0.0 for _ in self.variants]
        abs_errors = [0.0 for _ in self.variants]
        counts = [0.0 for _ in self.variants]

        for variant in self.variants:
            variant["model"].eval()

        with torch.no_grad():
            for _, (x, m, y) in enumerate(valid_loader):
                x = x.to(self.device)
                m = m.to(self.device)
                target = y[:, :, :1, :].to(self.device)
                observed_mask = m[:, :, :1, :].to(self.device)

                for idx, variant in enumerate(self.variants):
                    preds = variant["model"](x, m)
                    diff = (preds - target) * observed_mask
                    sq_errors[idx] += torch.sum(diff ** 2).item()
                    abs_errors[idx] += torch.sum(torch.abs(diff)).item()
                    counts[idx] += torch.sum(observed_mask).item()

        rmse_values = []
        mae_values = []
        for idx in range(len(self.variants)):
            denom = max(counts[idx], self.eps)
            rmse = float(np.sqrt(sq_errors[idx] / denom))
            mae = float(abs_errors[idx] / denom)
            rmse_values.append(rmse)
            mae_values.append(mae)

        val_tensor = torch.tensor(rmse_values, dtype=torch.float32)
        return val_tensor, mae_values

    def _update_weights(self, val_scores):
        self.accumulated_scores += val_scores
        weights = torch.softmax(-self.delta * self.accumulated_scores, dim=0)
        self.current_weights = weights
        ensemble_score = torch.sum(weights * val_scores).item()

        improved = False
        if ensemble_score + 1e-8 < self.best_ensemble_score:
            self.best_ensemble_score = ensemble_score
            self.best_weights = weights.clone()
            self.best_states = [
                self._snapshot_state(variant["model"]) for variant in self.variants
            ]
            self.best_epoch = self.completed_epochs
            self.patience = 0
            improved = True
        else:
            self.patience += 1

        self.history["val_rmse"].append(val_scores.tolist())
        self.history["weights"].append(weights.tolist())
        return ensemble_score, improved

    def train(self, train_loader, valid_loader, epochs):
        for epoch in range(epochs):
            start_time = time.time()
            train_losses = self._train_one_epoch(train_loader)
            val_scores, val_mae = self._evaluate_validation(valid_loader)
            self.completed_epochs = epoch + 1
            ensemble_score, improved = self._update_weights(val_scores)
            self.history["train_loss"].append(train_losses)
            epoch_time = time.time() - start_time

            self._log_epoch(epoch, train_losses, val_scores, val_mae, ensemble_score, epoch_time, improved)

            if self.patience > self.patience_limit:
                print("[UDCGLEnsemble] Early Stop triggered due to no ensemble improvement.")
                break

        if self.best_states is None:
            self.best_states = [self._snapshot_state(variant["model"]) for variant in self.variants]
            self.best_weights = self.current_weights.clone()

    def _log_epoch(
        self,
        epoch_idx,
        train_losses,
        val_scores,
        val_mae,
        ensemble_score,
        epoch_time,
        improved,
    ):
        acc_np = self.accumulated_scores.detach().cpu().numpy()
        weights_np = self.current_weights.detach().cpu().numpy()
        val_np = val_scores.detach().cpu().numpy()
        print(f"[UDCGLEnsemble] Epoch {epoch_idx + 1}")
        for idx, variant in enumerate(self.variants):
            print(
                "  {} | train loss = {:.6f} | S^t(n) = {:.6f} | A^t(n) = {:.6f} | ε^t(n) = {:.6f} | Val MAE = {:.6f}".format(
                    variant["name"],
                    train_losses[idx],
                    val_np[idx],
                    acc_np[idx],
                    weights_np[idx],
                    val_mae[idx],
                )
            )
        print(
            "  Ensemble RMSE = {:.6f} | Patience {}/{} | Duration {:.2f}s".format(
                ensemble_score, self.patience, self.patience_limit, epoch_time
            )
        )
        if improved:
            print(
                "    ✓ Improved ensemble performance. Snapshotting states at epoch {}.".format(
                    epoch_idx + 1
                )
            )

    def save_best_models(self, save_root, dataset, miss_mechanism, miss_ratio, seed):
        self.restore_best_states()
        target_dir = os.path.join(save_root, f"{dataset}")
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        saved_paths = []
        for variant, state in zip(self.variants, self.best_states):
            path = os.path.join(
                target_dir,
                f"{variant['name']}_best_model_{miss_mechanism}_ms{miss_ratio}_seed{seed}.pth",
            )
            torch.save(state, path)
            saved_paths.append(path)

        weight_path = os.path.join(
            target_dir,
            f"ensemble_weights_{miss_mechanism}_ms{miss_ratio}_seed{seed}.pt",
        )
        torch.save(self.best_weights, weight_path)
        print("[UDCGLEnsemble] Saved best checkpoints:")
        for path in saved_paths:
            print(f"  - {path}")
        print(f"  - Adaptive weight trace: {weight_path}")
        return saved_paths, weight_path

    def predict_ensemble(
        self,
        test_loader,
        mean,
        std,
        dataset,
        miss_ratio,
        miss_mechanism,
        seed,
        result_root,
        collect_outputs=False,
        weight_override=None,
    ):
        self.restore_best_states()
        weights = self.best_weights.detach().cpu().numpy()
        if weight_override is not None:
            override = np.asarray(weight_override, dtype=np.float32)
            override = override.reshape(-1)
            if len(override) != len(self.variants):
                raise ValueError(
                    "weight_override length {} does not match number of variants {}".format(
                        len(override), len(self.variants)
                    )
                )
            total = float(np.sum(override))
            if total <= 0:
                raise ValueError("weight_override must have a positive sum for normalisation")
            weights = (override / total).astype(np.float32)

        test_maes, test_rmses, test_mapes = [], [], []
        variant_metrics = {
            variant["name"]: {"mae": [], "rmse": [], "mape": []}
            for variant in self.variants
        }
        miss_data, predict_results, groundtruths = [], [], []

        if collect_outputs:
            def _init_stats():
                return {
                    "sum": 0.0,
                    "sq_sum": 0.0,
                    "count": 0,
                    "min": float("inf"),
                    "max": float("-inf"),
                }

            missing_stats = {variant["name"]: _init_stats() for variant in self.variants}
            full_stats = {variant["name"]: _init_stats() for variant in self.variants}

        for variant in self.variants:
            variant["model"].eval()

        with torch.no_grad():
            for _, (x, m, y) in enumerate(test_loader):
                x = x.to(self.device)
                m = m.to(self.device)

                preds_per_variant = []
                for variant in self.variants:
                    preds = variant["model"](x, m)
                    preds_per_variant.append(preds.detach().cpu().numpy())

                stacked_preds = np.stack(preds_per_variant, axis=0)
                weight_view = weights.reshape(-1, 1, 1, 1, 1)
                ensemble_pred = np.sum(stacked_preds * weight_view, axis=0)

                x_input = x[:, :, :1, :].detach().cpu().numpy()
                y_target = y[:, :, :1, :].detach().cpu().numpy()
                mask = m[:, :, :1, :].detach().cpu().numpy()
                if collect_outputs:
                    missing_bool = (1.0 - mask).astype(bool)
                else:
                    missing_bool = None

                unnorm_x = unnormalization(x_input, mean, std)
                unnorm_y = unnormalization(y_target, mean, std)
                unnorm_ensemble = unnormalization(ensemble_pred, mean, std)

                mae, rmse, mape = observed_eval_np(unnorm_ensemble, unnorm_y, mask)
                test_maes.append(mae)
                test_rmses.append(rmse)
                test_mapes.append(mape)

                for idx, variant in enumerate(self.variants):
                    unnorm_variant = unnormalization(stacked_preds[idx], mean, std)
                    v_mae, v_rmse, v_mape = observed_eval_np(unnorm_variant, unnorm_y, mask)
                    variant_metrics[variant["name"]]["mae"].append(v_mae)
                    variant_metrics[variant["name"]]["rmse"].append(v_rmse)
                    variant_metrics[variant["name"]]["mape"].append(v_mape)
                    if collect_outputs:
                        flat_vals = unnorm_variant.reshape(-1)
                        stats = full_stats[variant["name"]]
                        stats["sum"] += float(np.sum(flat_vals))
                        stats["sq_sum"] += float(np.sum(flat_vals ** 2))
                        stats["count"] += flat_vals.size
                        stats["min"] = min(stats["min"], float(np.min(flat_vals)))
                        stats["max"] = max(stats["max"], float(np.max(flat_vals)))
                        if missing_bool is not None and np.any(missing_bool):
                            miss_vals = unnorm_variant[missing_bool].reshape(-1)
                            miss_stats = missing_stats[variant["name"]]
                            miss_stats["sum"] += float(np.sum(miss_vals))
                            miss_stats["sq_sum"] += float(np.sum(miss_vals ** 2))
                            miss_stats["count"] += miss_vals.size
                            miss_stats["min"] = min(
                                miss_stats["min"],
                                float(np.min(miss_vals)) if miss_vals.size else miss_stats["min"],
                            )
                            miss_stats["max"] = max(
                                miss_stats["max"],
                                float(np.max(miss_vals)) if miss_vals.size else miss_stats["max"],
                            )

                predict_data = unnorm_ensemble * (1 - mask) + unnorm_x * mask
                unnorm_x = np.where(mask == 0, np.nan, unnorm_x)

                miss_data.append(unnorm_x)
                predict_results.append(predict_data)
                groundtruths.append(unnorm_y)

        test_mae = float(np.mean(test_maes))
        test_rmse = float(np.mean(test_rmses))
        test_mape = float(np.mean(test_mapes))
        print(
            "[UDCGLEnsemble] Test ensemble metrics — MAE {:.6f}, RMSE {:.6f}, MAPE {:.6f}".format(
                test_mae, test_rmse, test_mape * 100
            )
        )

        aggregated_variant_metrics = {}
        for name, metric_dict in variant_metrics.items():
            aggregated_variant_metrics[name] = {
                "mae": float(np.mean(metric_dict["mae"])) if metric_dict["mae"] else float("nan"),
                "rmse": float(np.mean(metric_dict["rmse"])) if metric_dict["rmse"] else float("nan"),
                "mape": float(np.mean(metric_dict["mape"])) if metric_dict["mape"] else float("nan"),
            }
            print(
                "    {} individual metrics — MAE {:.6f}, RMSE {:.6f}, MAPE {:.6f}".format(
                    name,
                    aggregated_variant_metrics[name]["mae"],
                    aggregated_variant_metrics[name]["rmse"],
                    aggregated_variant_metrics[name]["mape"] * 100,
                )
            )

        print(
            "    Final adaptive weights ε^t(n): {}".format(
                ", ".join(
                    "{} = {:.6f}".format(self.variants[idx]["name"], weights[idx])
                    for idx in range(len(weights))
                )
            )
        )

        result = {
            "missed_data": np.concatenate(miss_data, axis=0),
            "imputed_data": np.concatenate(predict_results, axis=0),
            "groundtruth": np.concatenate(groundtruths, axis=0),
        }

        print(result["missed_data"].shape)
        print(result["imputed_data"].shape)
        print(result["groundtruth"].shape)

        result_path = os.path.join(result_root, f"{dataset}")
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        with open(
            os.path.join(
                result_path,
                "result_{}_ms{}_seed{}.pkl".format(miss_mechanism, miss_ratio, seed),
            ),
            "wb",
        ) as fb:
            pk.dump(result, fb)

        diagnostics = None
        if collect_outputs:
            diagnostics = {"variant_gaussian_stats": {}}
            for variant in self.variants:
                name = variant["name"]
                target_stats = (
                    missing_stats[name]
                    if missing_stats[name]["count"] > 0
                    else full_stats[name]
                )
                count = target_stats["count"]
                if count > 0:
                    mean_val = target_stats["sum"] / count
                    variance = max(target_stats["sq_sum"] / count - mean_val ** 2, 0.0)
                    std_val = float(np.sqrt(variance))
                    min_val = (
                        target_stats["min"]
                        if np.isfinite(target_stats["min"])
                        else mean_val
                    )
                    max_val = (
                        target_stats["max"]
                        if np.isfinite(target_stats["max"])
                        else mean_val
                    )
                else:
                    mean_val = float("nan")
                    std_val = float("nan")
                    min_val = float("nan")
                    max_val = float("nan")
                diagnostics["variant_gaussian_stats"][name] = {
                    "mean": float(mean_val),
                    "std": std_val,
                    "min": float(min_val),
                    "max": float(max_val),
                    "count": int(count),
                    "source": "missing" if missing_stats[name]["count"] > 0 else "all_predictions",
                }
            diagnostics["ensemble_weights"] = weights.tolist()
            diagnostics["ensemble_metrics"] = {
                "mae": test_mae,
                "rmse": test_rmse,
                "mape": test_mape,
            }
            diagnostics["variant_metrics"] = {
                name: {
                    "mae": aggregated_variant_metrics[name]["mae"],
                    "rmse": aggregated_variant_metrics[name]["rmse"],
                    "mape": aggregated_variant_metrics[name]["mape"],
                }
                for name in aggregated_variant_metrics
            }

        return diagnostics




def predict(
    framework,
    result_path,
    test_loader,
    mean,
    std,
    dataset,
    miss_ratio,
    miss_mechanism,
    seed,
    collect_outputs=False,
    weight_override=None,
):
    return framework.predict_ensemble(
        test_loader=test_loader,
        mean=mean,
        std=std,
        dataset=dataset,
        miss_ratio=miss_ratio,
        miss_mechanism=miss_mechanism,
        seed=seed,
        result_root=result_path,
        collect_outputs=collect_outputs,
        weight_override=weight_override,
    )


def main(args):
    if args.learnable == 1:
        learnable = True
    else:
        learnable = False
    config_filename = args.config_path
    with open(config_filename) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print(config)

    dataset = config["data"]["dataset"]
    miss_mechanism = config["data"]["miss_mechanism"]
    # miss_pattern = config["data"]["miss_pattern"]
    miss_ratio = float(config["data"]["miss_ratio"])
    if args.miss_ratio is not None:
        miss_ratio = args.miss_ratio
    batch_size = int(config["data"]["batch_size"])
    val_batch_size = int(config["data"]["val_batch_size"])
    test_batch_size = int(config["data"]["test_batch_size"])
    seqlen = int(config["model"]["seqlen"])
    num_nodes = int(config["model"]["num_nodes"])
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
    seed = args.seed

    device = torch.device(
        "cuda:{}".format(int(config["train"]["cuda"]))
        if torch.cuda.is_available()
        else "cpu"
    )

    seed_torch(seed)
    torch.set_num_threads(10)

    train_loader, valid_loader, test_loader, mean, std, A = generate_miss_loader(
        dataset,
        miss_mechanism,
        # miss_pattern,
        miss_ratio,
        seqlen,
        batch_size,
        val_batch_size,
        test_batch_size,
    )
    num_nodes = A.shape[0]

    # wandb.init(
    #     project="UDCGL",
    #     name="{}_lr{}_hiddensize{}_batchsize{}_seed{}".format(
    #         dataset, lr, hidden_size, batch_size, seed
    #     ),
    # )

    adj_mx = weight_matrix(A)

    log_strategy_steps = bool(config["train"].get("log_strategy_steps", True))

    base_model_kwargs = dict(
        device=device,
        num_nodes=num_nodes,
        seqlen=seqlen,
        in_channels=in_channel,
        hidden_dim=hidden_size,
        st_block=st_block,
        K=K,
        d_model=d_model,
        n_heads=n_heads,
        adj_mx=adj_mx,
        learnable=learnable,
        log_strategy_steps=log_strategy_steps,
    )

    ensemble_cfg = config["train"].get("ensemble", {})
    ensemble_delta = float(ensemble_cfg.get("delta", 0.5))
    default_lambda = float(ensemble_cfg.get("lambda", 1e-4))
    ensemble_l1_lambda = float(ensemble_cfg.get("lambda_l1", default_lambda))
    ensemble_l2_lambda = float(ensemble_cfg.get("lambda_l2", default_lambda))
    patience_limit = int(config["train"].get("patience", 20))

    framework = UDCGLEnsemble(
        device=device,
        base_model_kwargs=base_model_kwargs,
        lr=lr,
        delta=ensemble_delta,
        l1_lambda=ensemble_l1_lambda,
        l2_lambda=ensemble_l2_lambda,
        patience_limit=patience_limit,
        log_strategy_steps=log_strategy_steps,
    )

    framework.train(train_loader, valid_loader, epochs)

    if framework.best_epoch != -1:
        print(
            "[UDCGLEnsemble] Best ensemble epoch: {} (ensemble RMSE {:.6f})".format(
                framework.best_epoch, framework.best_ensemble_score
            )
        )

    framework.save_best_models(
        save_path,
        dataset,
        miss_mechanism,
        miss_ratio,
        seed,
    )

    # test
    predict(
        framework,
        result_path,
        test_loader,
        mean,
        std,
        dataset,
        miss_ratio,
        miss_mechanism,
        seed,
    )


if __name__ == "__main__":
    start_time = time.time()
    main(args)
    print("Spend Time: {}".format(time.time() - start_time))
