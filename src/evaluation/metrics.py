"""
All evaluation metrics, shared by every model (numpy / pandas / sklearn only).

Regression (per target, SBP and DBP):
  MAE, RMSE, Pearson r, R^2
  ME (bias) and SDE (SD of error)            -> AAMI / ISO 81060-2 criterion: |ME| <= 5 and SDE <= 8 mmHg
  Bland-Altman 95% limits of agreement       -> ME +/- 1.96 SDE
  % |error| <= 5 / 10 / 15 mmHg and grade    -> BHS protocol (A: 60/85/95, B: 50/75/90, C: 40/65/85)
  MAE grade                                  -> IEEE 1708 (A <= 5, B 5-6, C 6-7, D > 7 mmHg)
  within-subject correlation (median, IQR)   -> BP tracking within a person (project-specific)
Grades are reported descriptively (this is not a formal clinical validation protocol).

Classification (from predicted SBP/DBP via the same thresholds as the labels):
  confusion matrix, accuracy, balanced accuracy, macro-F1, quadratic-weighted Cohen's kappa,
  per-class sensitivity / specificity / precision / F1; binary task also AUROC / AUPRC.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             cohen_kappa_score, confusion_matrix, f1_score, roc_auc_score)

TARGETS = ["sbp", "dbp"]


# ----------------------------------------------------------------------------- regression
def bhs_grade(w5, w10, w15):
    for g, (a, b, c) in {"A": (60, 85, 95), "B": (50, 75, 90), "C": (40, 65, 85)}.items():
        if w5 * 100 >= a and w10 * 100 >= b and w15 * 100 >= c:
            return g
    return "D"


def ieee1708_grade(mae):
    return "A" if mae <= 5 else "B" if mae <= 6 else "C" if mae <= 7 else "D"


def _pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def within_subject_r(df, t, min_rec=5):
    rs = [_pearson(g[t], g[f"{t}_pred"]) for _, g in df.groupby("pid") if len(g) >= min_rec]
    rs = np.array([r for r in rs if np.isfinite(r)])
    if len(rs) == 0:
        return float("nan"), float("nan"), float("nan"), 0
    return float(np.median(rs)), float(np.percentile(rs, 25)), float(np.percentile(rs, 75)), int(len(rs))


def regression_metrics(df, targets=TARGETS, min_rec=5):
    out = {"n": int(len(df)), "subjects": int(df["pid"].nunique())}
    for t in targets:
        y, p = df[t].to_numpy(float), df[f"{t}_pred"].to_numpy(float)
        e = p - y
        ae = np.abs(e)
        me, sde = float(e.mean()), float(e.std(ddof=1)) if len(e) > 1 else float("nan")
        w5, w10, w15 = (float((ae <= k).mean()) for k in (5, 10, 15))
        mae = float(ae.mean())
        ss_res, ss_tot = float((e ** 2).sum()), float(((y - y.mean()) ** 2).sum())
        ws_med, ws_q1, ws_q3, ws_n = within_subject_r(df, t, min_rec)
        out[t] = {
            "MAE": mae, "RMSE": float(np.sqrt((e ** 2).mean())), "r": _pearson(y, p),
            "R2": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
            "ME": me, "SDE": sde, "LoA_low": me - 1.96 * sde, "LoA_high": me + 1.96 * sde,
            "within5": w5, "within10": w10, "within15": w15,
            "BHS_grade": bhs_grade(w5, w10, w15), "IEEE1708_grade": ieee1708_grade(mae),
            "AAMI_pass": bool(abs(me) <= 5 and sde <= 8),
            "within_subject_r_median": ws_med, "within_subject_r_q1": ws_q1,
            "within_subject_r_q3": ws_q3, "within_subject_n": ws_n,
        }
    return out


def reference_baselines(df, train_df, targets=TARGETS, min_rec=5):
    """Training-mean predictor and subject-mean oracle (uses true labels: reference only)."""
    res = {}
    base = df.copy()
    for t in targets:
        base[f"{t}_pred"] = float(train_df[t].mean())
    res["baseline_training_mean"] = regression_metrics(base, targets, min_rec)
    orc = df.copy()
    for t in targets:
        orc[f"{t}_pred"] = df.groupby("pid")[t].transform("mean")
    res["reference_subject_mean_oracle"] = regression_metrics(orc, targets, min_rec)
    return res


# ----------------------------------------------------------------------------- classification
CLASS4 = ["normal", "elevated", "stage1", "stage2"]


def bp_class4(sbp, dbp):
    sbp, dbp = np.asarray(sbp, float), np.asarray(dbp, float)
    return np.select([(sbp >= 140) | (dbp >= 90), (sbp >= 130) | (dbp >= 80), sbp >= 120],
                     ["stage2", "stage1", "elevated"], default="normal")


SCHEMES = {
    "4class": (CLASS4, lambda c: c),
    "3class": (["normal_elevated", "stage1", "stage2"],
               lambda c: np.where(np.isin(c, ["normal", "elevated"]), "normal_elevated", c)),
    "binary_130_80": (["below", "high"], lambda c: np.where(np.isin(c, ["stage1", "stage2"]), "high", "below")),
}


def _per_class(cm, labels):
    out = {}
    total = cm.sum()
    for i, lab in enumerate(labels):
        tp = cm[i, i]; fn = cm[i].sum() - tp; fp = cm[:, i].sum() - tp; tn = total - tp - fn - fp
        sens = tp / (tp + fn) if tp + fn else float("nan")
        prec = tp / (tp + fp) if tp + fp else float("nan")
        out[lab] = {"support": int(tp + fn), "sensitivity": float(sens),
                    "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
                    "precision": float(prec),
                    "f1": float(2 * prec * sens / (prec + sens)) if (prec + sens) and np.isfinite(prec + sens) else 0.0}
    return out


def classification_metrics(df):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")      # sklearn warns when a class is absent in a small subgroup
        return _classification_metrics(df)


def _classification_metrics(df):
    true4 = bp_class4(df["sbp"], df["dbp"])
    pred4 = bp_class4(df["sbp_pred"], df["dbp_pred"])
    res = {}
    for name, (labels, f) in SCHEMES.items():
        t, p = f(true4), f(pred4)
        ti = np.array([labels.index(v) for v in t]); pi = np.array([labels.index(v) for v in p])
        cm = confusion_matrix(ti, pi, labels=list(range(len(labels))))
        r = {"labels": labels, "confusion_matrix": cm.tolist(),
             "accuracy": float(accuracy_score(ti, pi)),
             "balanced_accuracy": float(balanced_accuracy_score(ti, pi)),
             "macro_f1": float(f1_score(ti, pi, average="macro", labels=list(range(len(labels))), zero_division=0)),
             "kappa": float(cohen_kappa_score(ti, pi, weights="quadratic" if len(labels) > 2 else None)),
             "per_class": _per_class(cm, labels)}
        if name == "binary_130_80":
            # continuous score: how far the predicted BP is above the 130/80 threshold (mmHg)
            score = np.maximum(df["sbp_pred"].to_numpy() - 130, df["dbp_pred"].to_numpy() - 80)
            if 0 < ti.sum() < len(ti):
                r["auroc"] = float(roc_auc_score(ti, score))
                r["auprc"] = float(average_precision_score(ti, score))
        res[name] = r
    return res


# ----------------------------------------------------------------------------- full evaluation
def evaluate_predictions(pred, train_df=None, group_cols=("protocol", "phase", "is_exercise"), min_rec=5):
    res = {"overall": regression_metrics(pred, TARGETS, min_rec),
           "classification": classification_metrics(pred)}
    for col in group_cols:
        if col in pred:
            res[f"by_{col}"] = {str(k): regression_metrics(g, TARGETS, min_rec)
                                for k, g in pred.groupby(col) if len(g) >= 2}
    if train_df is not None:
        res.update(reference_baselines(pred, train_df, TARGETS, min_rec))
    return res
