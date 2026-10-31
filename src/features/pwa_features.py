"""
Handcrafted ECG/PPG pulse-wave features used by HGCTNet and BpBLS re-implementations.

Definitions: Liu et al., IEEE JBHI 27(9), 2023, Table A1 and Fig. 3; K value: Miao et al., IEEE JBHI 21(6), 2017.
Specification and deviations: docs/reimplementation/HGCTNet.md (sections 4 and 8).

Input: one processed window (ECG and PPG, same length, fs Hz; ECG R peaks upright, PPG upright, z-scored).
Per cardiac cycle (PPG foot to next foot) the characteristic points are detected
  PPG : s foot (intersecting tangent), m maximum upslope, p systolic peak, n dicrotic notch, v end of cycle (= next foot)
  VPPG: s minimum before the main peak, p main peak (= m of the PPG), v minimum after the peak
  APPG: s minimum before the main peak, p main peak (a-wave), v minimum after the peak (b-wave)
features are computed per cycle and averaged over the valid cycles of the window (31 features).
"""
import numpy as np
from scipy.signal import find_peaks

FEATURE_NAMES = (
    ["ptt_rs1", "ptt_rm1", "ptt_rp1"]
    + [f"{f}_{s}" for f in ("asc_time", "desc_time", "asc_slope", "desc_slope", "asc_area", "desc_area", "aid", "did")
       for s in ("ppg", "vppg", "appg")]
    + ["lasi_ppg", "pulse_width_ppg", "pulse_rate_ppg", "k_value_ppg"]
)
assert len(FEATURE_NAMES) == 31


def _derivatives(ppg, fs):
    v = np.gradient(ppg) * fs
    a = np.gradient(v) * fs
    return v, a


def detect_points(ecg, ppg, fs=125):
    """Characteristic points per cycle. Returns (list of dicts, R-peak indices, VPPG, APPG)."""
    ecg = np.asarray(ecg, float)
    ppg = np.asarray(ppg, float)
    v1, a = _derivatives(ppg, fs)
    n = len(ppg)
    # ECG R peaks
    r_thr = 0.5 * np.percentile(ecg, 99.5)
    rpk, _ = find_peaks(ecg, height=max(r_thr, 0.5), distance=int(0.33 * fs))
    # PPG maximum upslope points (peaks of the first derivative)
    m_thr = 0.4 * np.percentile(v1, 99)
    mpk, _ = find_peaks(v1, height=m_thr, distance=int(0.33 * fs))
    if len(mpk) < 2:
        return [], rpk, v1, a
    # feet (s): intersecting-tangent point - the tangent at the maximum upslope meets the horizontal line through
    # the minimum of the PPG in the stretch before the upslope (bounded by the previous upslope); this is the
    # standard PPG foot and coincides with the trough when the diastolic decay reaches the upstroke
    feet = []
    for i, m in enumerate(mpk):
        lo = max(0, m - int(0.4 * fs), mpk[i - 1] + int(0.15 * fs) if i > 0 else 0)
        if m - lo < 2 or v1[m] <= 0:
            feet.append(-1)
            continue
        y0 = ppg[lo:m].min()
        foot = int(round(m - (ppg[m] - y0) / v1[m] * fs))
        feet.append(int(np.clip(foot, lo, m - 1)))
    cycles = []
    for i in range(len(mpk) - 1):
        s, m, nxt = feet[i], mpk[i], feet[i + 1]
        if s < 0 or nxt < 0 or nxt <= m:
            continue
        v = nxt
        T = (v - s) / fs
        if not (0.33 <= T <= 2.0):                          # 30-180 bpm
            continue
        hi = min(m + int(0.35 * fs), v - 1)
        if hi <= m:
            continue
        p = m + int(np.argmax(ppg[m:hi + 1]))
        # dicrotic notch: first local minimum of the PPG after the peak; fallback = APPG maximum (inflection)
        lo_n, hi_n = p + int(0.05 * fs), p + int(0.6 * (v - p))
        notch, fallback = -1, False
        if hi_n - lo_n >= 3:
            seg = ppg[lo_n:hi_n + 1]
            # a notch must be a real dip (prominence >= 2% of the pulse amplitude), not a ripple on a flat peak
            mins, _ = find_peaks(-seg, prominence=0.02 * max(ppg[p] - ppg[s], 1e-6))
            if len(mins):
                notch = lo_n + int(mins[0])
            else:
                notch = lo_n + int(np.argmax(a[lo_n:hi_n + 1]))
                fallback = True
        # VPPG points
        vp = m
        v_lo = max(0, s - int(0.15 * fs), cycles[-1]["vp"] + 1 if cycles else 0)   # start of the VPPG rise
        vs = v_lo + int(np.argmin(v1[v_lo:vp + 1])) if vp > v_lo else v_lo
        vv = vp + int(np.argmin(v1[vp:v + 1]))
        # APPG points: a-wave peak before the maximum upslope, b-wave minimum after it
        a_lo = max(0, s - int(0.1 * fs))
        ap = a_lo + int(np.argmax(a[a_lo:m + 1]))
        as_ = a_lo + int(np.argmin(a[a_lo:ap + 1])) if ap > a_lo else a_lo
        av_hi = min(v, ap + int(0.4 * (v - s)))
        av = ap + int(np.argmin(a[ap:av_hi + 1])) if av_hi > ap else ap
        # preceding R peak (pulse arrival)
        prior = rpk[(rpk < m - int(0.05 * fs)) & (rpk > m - int(0.8 * fs))]
        r = int(prior[-1]) if len(prior) else -1
        cycles.append({"s": s, "m": m, "p": p, "n": notch, "v": v, "notch_fallback": fallback,
                       "vs": vs, "vp": vp, "vv": vv, "as": as_, "ap": ap, "av": av, "r": r})
    return cycles, rpk, v1, a


def _seg_feats(x, s, p, v, fs):
    if not (s < p < v):
        return [np.nan] * 8
    ta, td = (p - s) / fs, (v - p) / fs
    aid, did = x[p] - x[s], x[p] - x[v]
    return [ta, td, aid / ta, (x[v] - x[p]) / td,
            float(np.sum(x[s:p + 1] - x[s]) / fs), float(np.sum(x[p:v + 1] - x[v]) / fs), aid, did]


def cycle_features(c, ppg, v1, a, fs):
    """31 features of one cycle (NaN where a point is missing)."""
    r = c["r"]
    ptt = [(c[k] - r) / fs if r >= 0 and c[k] > r else np.nan for k in ("s", "m", "p")]
    if r >= 0 and not (0.0 < ptt[0] <= 0.6):                 # implausible pulse arrival: discard PTTs
        ptt = [np.nan] * 3
    seg = {"ppg": _seg_feats(ppg, c["s"], c["p"], c["v"], fs),
           "vppg": _seg_feats(v1, c["vs"], c["vp"], c["vv"], fs),
           "appg": _seg_feats(a, c["as"], c["ap"], c["av"], fs)}
    per = []
    for j in range(8):                                          # order: feature type outer, signal inner
        for sname in ("ppg", "vppg", "appg"):
            per.append(seg[sname][j])
    x = ppg[c["s"]:c["v"] + 1]
    rng = x.max() - x.min()
    lasi = (c["n"] - c["p"]) / fs if c["n"] > c["p"] else np.nan
    pw = (c["n"] - c["m"]) / fs if c["n"] > c["m"] else np.nan
    return ptt + per + [lasi, pw, 60.0 / ((c["v"] - c["s"]) / fs), (x.mean() - x.min()) / rng if rng > 0 else np.nan]


def extract_features(ecg, ppg, fs=125, min_cycles=2):
    """Mean of the per-cycle features over the window; NaN if fewer than min_cycles valid cycles.
    Returns (feature vector of length 31, n_cycles, notch_fallback_fraction)."""
    ppg = np.asarray(ppg, float)
    cycles, _, v1, a = detect_points(ecg, ppg, fs)
    if len(cycles) < min_cycles:
        return np.full(31, np.nan), len(cycles), np.nan
    F = np.array([cycle_features(c, ppg, v1, a, fs) for c in cycles], float)
    with np.errstate(all="ignore"):
        feats = np.nanmean(F, axis=0)
    return feats, len(cycles), float(np.mean([c["notch_fallback"] for c in cycles]))
