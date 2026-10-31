"""
Generic trainer used by every model.

  * AdamW + OneCycle LR (stepped per batch), gradient clipping, bf16 autocast
  * EMA weights (evaluated and checkpointed when ema_decay > 0)
  * DropPath warm-up through model.set_drop_path(scale) if the model provides it
  * early stopping on validation MAE in mmHg (mean of SBP and DBP)
  * checkpoint = weights + full resolved config + target statistics -> model can always be rebuilt
Test data is never touched here; the training script predicts it once, after training.
"""
import time

import numpy as np
import pandas as pd
import torch

from .ema import ModelEMA
from .losses import main_loss, masked_aux_loss


class Trainer:
    def __init__(self, cfg, model, train_loader, val_loader, y_val_mmhg, stats, out_dir, log, device):
        self.cfg, self.model, self.log, self.device = cfg, model, log, device
        self.train_loader, self.val_loader = train_loader, val_loader
        self.y_val = np.asarray(y_val_mmhg, dtype=np.float64)
        self.stats, self.out_dir = stats, out_dir
        t = cfg["train"]
        self.epochs = int(t["epochs"])
        self.opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
        total_steps = self.epochs * len(train_loader)
        pct_start = min(0.5, max(float(t["pct_start"]), 2.0 / total_steps))   # OneCycle needs >= 2 warm-up steps
        self.sched = torch.optim.lr_scheduler.OneCycleLR(
            self.opt, max_lr=t["lr"], total_steps=total_steps,
            pct_start=pct_start, div_factor=t["div_factor"], final_div_factor=t["final_div_factor"])
        self.ema = ModelEMA(model, t["ema_decay"]) if t.get("ema_decay", 0) > 0 else None
        self.use_amp = bool(t.get("amp_bf16", True)) and device.type == "cuda"
        self.aux_w = float(cfg["loss"].get("aux_weight", 0.0))

    # ------------------------------------------------------------------ helpers
    def eval_model(self):
        return self.ema.module if self.ema is not None else self.model

    def _forward(self, model, b):
        """Model call for single-recording batches and calibration-pair batches (x_cal, bp_cal)."""
        kw = {k: b[k].to(self.device, non_blocking=True) for k in ("x_cal", "bp_cal", "ctx", "demo") if k in b}
        return model(b["x"].to(self.device, non_blocking=True), **kw)

    def _autocast(self):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.use_amp)

    @torch.no_grad()
    def predict(self, loader, model=None):
        """Returns dict of numpy arrays in loader order: bp (mmHg), and aux / gates if the model outputs them."""
        model = model or self.eval_model()
        model.eval()
        outs = {"bp": [], "aux": [], "gates": []}
        for b in loader:
            with self._autocast():
                o = self._forward(model, b)
            for k in outs:
                if k in o and o[k] is not None:
                    outs[k].append(o[k].float().cpu().numpy())
        res = {k: np.concatenate(v) for k, v in outs.items() if v}
        res["bp"] = res["bp"] * self.stats["y_std"] + self.stats["y_mean"]
        if "aux" in res and len(self.stats.get("aux_mean", [])):
            res["aux"] = res["aux"] * self.stats["aux_std"] + self.stats["aux_mean"]
        return res

    def _set_drop_path(self, epoch):
        fn = getattr(self.model, "set_drop_path", None)
        if callable(fn):
            warm = int(self.cfg["train"].get("drop_path_warmup_epochs", 0))
            fn(min(1.0, epoch / warm) if warm > 0 else 1.0)

    def save_checkpoint(self, path, epoch, val_mae):
        torch.save({"model_state": self.eval_model().state_dict(), "config": self.cfg, "epoch": epoch,
                    "val_mae": val_mae, "stats": {k: np.asarray(v).tolist() for k, v in self.stats.items()},
                    "ema": self.ema is not None}, path)

    # ------------------------------------------------------------------ training
    def train_one_epoch(self, epoch):
        self.model.train()
        self._set_drop_path(epoch)
        lc = self.cfg["loss"]
        tot, tot_main, tot_aux, n = 0.0, 0.0, 0.0, 0
        for b in self.train_loader:
            y = b["y"].to(self.device, non_blocking=True)
            with self._autocast():
                o = self._forward(self.model, b)
            l_main = main_loss(o["bp"], y, lc["main"], lc.get("huber_delta", 1.0))
            loss = l_main
            l_aux = torch.zeros((), device=self.device)
            if self.aux_w > 0 and "aux" in o:
                l_aux = masked_aux_loss(o["aux"], b["aux"].to(self.device), b["aux_mask"].to(self.device),
                                        lc.get("huber_delta", 1.0))
                loss = loss + self.aux_w * l_aux
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}")
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["train"]["grad_clip"])
            self.opt.step()
            self.sched.step()
            if self.ema is not None:
                self.ema.update(self.model)
            tot += loss.item(); tot_main += l_main.item(); tot_aux += l_aux.item(); n += 1
        return tot / n, tot_main / n, tot_aux / n

    def fit(self):
        t = self.cfg["train"]
        best, best_epoch, wait, hist = np.inf, 0, 0, []
        ckpt = self.out_dir / "best.pt"
        t0 = time.time()
        for ep in range(1, self.epochs + 1):
            te = time.time()
            loss, l_main, l_aux = self.train_one_epoch(ep)
            pv = self.predict(self.val_loader)["bp"]                       # EMA weights (if enabled)
            mae = np.abs(pv - self.y_val).mean(axis=0)
            score = float(mae.mean())
            if self.ema is not None:                                       # diagnostic: live weights too
                pl = self.predict(self.val_loader, model=self.model)["bp"]
                mae_live = np.abs(pl - self.y_val).mean(axis=0)
            else:
                pl, mae_live = pv, mae
            improved = score < best
            if improved:
                best, best_epoch, wait = score, ep, 0
                self.save_checkpoint(ckpt, ep, score)
            else:
                wait += 1
            hist.append({"epoch": ep, "loss": loss, "loss_main": l_main, "loss_aux": l_aux,
                         "val_sbp_mae": mae[0], "val_dbp_mae": mae[1], "val_score": score,
                         "val_live_sbp_mae": mae_live[0], "val_live_dbp_mae": mae_live[1],
                         "val_pred_mean_sbp": float(pv[:, 0].mean()), "val_pred_sd_sbp": float(pv[:, 0].std()),
                         "lr": self.sched.get_last_lr()[0], "epoch_s": time.time() - te})
            pd.DataFrame(hist).to_csv(self.out_dir / "history.csv", index=False)
            self.log.info(f"epoch {ep:3d} | loss {loss:.4f} (main {l_main:.4f}, aux {l_aux:.4f}) | "
                          f"val MAE SBP {mae[0]:.2f} DBP {mae[1]:.2f} | live SBP {mae_live[0]:.2f} "
                          f"DBP {mae_live[1]:.2f} | pred SBP mean {pv[:, 0].mean():.1f} sd {pv[:, 0].std():.1f} "
                          f"(true {self.y_val[:, 0].mean():.1f} sd {self.y_val[:, 0].std():.1f}) | {time.time() - te:.0f}s"
                          f"{'  *best*' if improved else ''}")
            if wait >= t["patience"]:
                self.log.info(f"early stopping: no improvement for {t['patience']} epochs")
                break
        self.log.info(f"best epoch {best_epoch} | val score {best:.3f} | {(time.time() - t0) / 60:.1f} min")
        state = torch.load(ckpt, map_location=self.device)["model_state"]
        self.eval_model().load_state_dict(state)
        return best_epoch, best
