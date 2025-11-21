"""
PIV analysis utilities for banded epithelial monolayers.

This module:
  * reads MATLAB .mat files for each band (one folder per band length),
  * extracts spatial/temporal periods and velocity metrics from PIV,
  * stores them in a cache file ('Sample-data.plk'),
  * generates summary plots and statistics from the cached data.

Once 'Sample-data.pkl' has been created from the raw .mat files, the script
can be run in "cache-only" mode so that all numbers are guaranteed to be
identical from run to run.

Current metrics stored per trace:
  * s     : spatial period
  * t     : temporal period
  * v     : effective phase velocity (ds/dt)
  * rms   : RMS velocity (space- and time-averaged)
  * vxpk  : robust peak |Vx| (P95 over space, then averaged over time)
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt

from typing import Dict, List, Tuple, Any
from scipy.io import loadmat as sci_loadmat
import h5py
import pickle
from math import sqrt

# -------------------- cache schema / versioning -----------------------------

# Increment this if the structure of recs/pool changes.
CACHE_VERSION = 3

# Keys that must be present in both the per-band records ("recs")
# and the pooled dictionary ("pool") for each trace.
REQUIRED_KEYS = ('s', 't', 'v', 'rms', 'vxpk')

# Legacy keys from older versions of the script that are now removed.
LEGACY_KEYS = ('j_s', 'j_t', 'h')

# Name of the cache file that will be (re)used by default.
# It is expected to live in the same directory as this script.
CACHE_FILENAME = "Sample-data.pkl"

try:
    import mat73
    HAVE_MAT73 = True
except ImportError:
    HAVE_MAT73 = False


# ---------- path helpers -----------------------------------------------------

def get_cache_path() -> str:
    """
    Return the absolute path of the cache file.

    By default, this is 'Sample-data.pkl' located in the same directory as
    this script. If __file__ is not defined (e.g. interactive session),
    the current working directory is used instead.
    """
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # Fallback for interactive / notebook-style execution
        base_dir = os.getcwd()
    return os.path.join(base_dir, CACHE_FILENAME)


# ---------- loader -----------------------------------------------------------

def load_mat(fname: str) -> dict:
    """
    Load a MATLAB .mat file in a reasonably robust way.

    The function tries, in order:
      1. mat73.loadmat (for v7.3 files, if available),
      2. scipy.io.loadmat (for "classic" MATLAB files),
      3. h5py (manual extraction from an HDF5 'Line' group).

    Parameters
    ----------
    fname : str
        Path to the .mat file.

    Returns
    -------
    dict
        A dictionary with at least a 'Line' key mimicking the MATLAB struct.
    """
    # Try mat73 for v7.3 HDF5-based .mat files.
    if HAVE_MAT73:
        try:
            return mat73.loadmat(fname)
        except Exception:
            pass

    # Fallback: standard SciPy loadmat.
    try:
        return sci_loadmat(fname, struct_as_record=False, squeeze_me=True)
    except Exception:
        pass

    # Last resort: treat it as a generic HDF5 file with a '/Line' group.
    with h5py.File(fname, "r") as f:
        line = {}
        for fld in f["/Line"].keys():
            # Flatten each dataset to a 1D array so it behaves nicely later
            line[fld] = np.asarray(f["/Line"][fld]).flatten()
        return {"Line": line}


# ---------- small helpers ----------------------------------------------------

def as_float(v: Any) -> float:
    """
    Convert an arbitrary value coming from MATLAB into a plain float.

    Handles:
      * numpy arrays (possibly 0D),
      * nested object dtypes,
      * and returns NaN if conversion fails.
    """
    try:
        arr = np.asarray(v).squeeze()
        if hasattr(arr, "dtype") and arr.dtype == object:
            # Some MATLAB structs end up as 0D arrays of dtype=object
            arr = arr.item()
        return float(arr)
    except Exception:
        return np.nan


def as_list(x: Any) -> List[float]:
    """
    Convert a variety of container-like objects to a Python list.

    Parameters
    ----------
    x : Any
        Could be None, numpy.ndarray, list, tuple, scalar...

    Returns
    -------
    list
        A Python list (possibly empty).
    """
    if x is None:
        return []
    if isinstance(x, np.ndarray):
        return x.flatten().tolist()
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def retrofit_cache(recs: Dict[float, Dict[str, List[float]]],
                   pool: Dict[str, List[float]]):
    """
    Upgrade older cache schemas to the current one.

    Operations:
      * drop legacy keys (j_s, j_t, h),
      * ensure all REQUIRED_KEYS are present in each band and in the pool.

    Parameters
    ----------
    recs : dict
        Per-band results, keyed by band length (float).
    pool : dict
        Pooled metrics over all bands.

    Returns
    -------
    recs, pool : dict, dict
        Updated dictionaries matching the current schema.
    """
    # Clean up per-band dictionaries.
    for L, d in recs.items():
        for lk in LEGACY_KEYS:
            d.pop(lk, None)
        for k in REQUIRED_KEYS:
            if k not in d:
                d[k] = []

    # Clean up pooled dictionary.
    for lk in LEGACY_KEYS:
        pool.pop(lk, None)
    for k in REQUIRED_KEYS:
        if k not in pool:
            pool[k] = []

    return recs, pool


# ---------- collect ----------------------------------------------------------

MetricLists = Dict[str, List[float]]


def _framewise_stats(vx: np.ndarray, vy: np.ndarray) -> Tuple[float, float]:
    """
    Compute time-averaged spatial RMS speed and robust peak |Vx|.

    For each time frame:
      * Compute sqrt( <vx^2 + vy^2>_space )  -> RMS speed per frame.
      * Compute P95(|vx|)                    -> "peak" speed per frame.

    Then average over time to obtain a single RMS and peak value per trace.

    Parameters
    ----------
    vx, vy : np.ndarray
        Velocity components from PIV. Can be 2D or 3D arrays.

    Returns
    -------
    rms_time : float
        Time-averaged spatial RMS speed.
    vxpk_time : float
        Time-averaged P95(|Vx|).
    """
    vx = np.asarray(vx, float)
    vy = np.asarray(vy, float)

    # Reshape to (n_frames, n_pixels) so that axis=1 is spatial.
    if vx.ndim >= 3:
        t = vx.shape[0]
        vx2d = vx.reshape(t, -1)
        vy2d = vy.reshape(t, -1)
    elif vx.ndim == 2:
        vx2d = vx.reshape(1, -1)
        vy2d = vy.reshape(1, -1)
    else:
        vx2d = vx.reshape(1, -1)
        vy2d = vy.reshape(1, -1)

    # Spatial RMS speed per frame, then time-average.
    frame_rms = np.sqrt(np.nanmean(vx2d**2 + vy2d**2, axis=1))  # shape (t,)
    rms_time = float(np.nanmean(frame_rms)) if frame_rms.size else np.nan

    # Robust "peak" |Vx| per frame (95th percentile), then time-average.
    frame_peak = np.nanpercentile(np.abs(vx2d), 95, axis=1)      # shape (t,)
    vxpk_time = float(np.nanmean(frame_peak)) if frame_peak.size else np.nan

    return rms_time, vxpk_time


def collect(base: str) -> Tuple[Dict[float, MetricLists], MetricLists]:
    """
    Scan a "base" directory for band subfolders and collect PIV metrics.

    Folder layout is assumed to be:

        base/
          100/
            .../data*.mat
          250/
            .../data*.mat
          ...

    For each 'data*.mat' file, the function expects a 'Line' struct containing
    'PIVPeaks' and 'PIV', and extracts:

      * s     : spatial period (from sPeriod / s_period),
      * t     : temporal period (from tPeriod / t_period),
      * v     : effective phase velocity (ds/dt from peaks),
      * rms   : RMS velocity over frames,
      * vxpk  : P95(|Vx|) over frames.

    Parameters
    ----------
    base : str
        Path to the directory containing the band subfolders.

    Returns
    -------
    recs : dict
        Per-band metrics, keyed by band length (float).
    pooled : dict
        Pooled metrics over all bands, keyed by metric name.
    """
    keys = ('s', 't', 'v', 'rms', 'vxpk')
    recs: Dict[float, MetricLists] = {}
    pooled = {k: [] for k in keys}

    # Select subdirectories that look like numeric band lengths (e.g. '100', '540.0').
    bands = sorted(
        [
            d for d in os.listdir(base)
            if os.path.isdir(os.path.join(base, d))
            and re.fullmatch(r"\d+(?:\.\d+)?", d)
        ],
        key=float
    )
    print("Bands:", bands)

    # Loop over bands.
    for b in bands:
        band_dir = os.path.join(base, b)
        lsts = {k: [] for k in keys}

        # Recursive walk inside each band directory to find 'data*.mat'.
        for root, _, files in os.walk(band_dir):
            for fn in files:
                if not fn.lower().startswith("data") or not fn.lower().endswith(".mat"):
                    continue

                fpath = os.path.join(root, fn)
                try:
                    line = load_mat(fpath)['Line']
                except Exception as e:
                    print(" skip", fn, ":", e)
                    continue

                # Only process files that contain both PIVPeaks and PIV.
                if isinstance(line, dict) and 'PIVPeaks' in line and 'PIV' in line:
                    peaks, vels = list(line['PIVPeaks']), list(line['PIV'])
                    for pk, vel in zip(peaks, vels):
                        if not isinstance(pk, dict):
                            continue

                        # Periods (spatial and temporal) extracted from MATLAB fields.
                        s_val = as_float(pk.get('sPeriod', pk.get('s_period', np.nan)))
                        t_val = as_float(pk.get('tPeriod', pk.get('t_period', np.nan)))

                        # RMS and peak |Vx| extracted from the PIV velocity fields.
                        vx, vy = vel.get('Vx'), vel.get('Vy')
                        if isinstance(vx, np.ndarray) and isinstance(vy, np.ndarray):
                            rms, vxpk = _framewise_stats(vx, vy)
                        else:
                            rms = np.nan
                            vxpk = np.nan

                        # Peak positions in space and time used to estimate phase velocity.
                        s_peaks = as_list(pk.get('sPeaks'))
                        t_peaks = as_list(pk.get('tPeaks'))

                        # Effective phase velocity ds/dt from consecutive peaks (if enough points exist).
                        v = np.nan
                        if len(s_peaks) > 1 and len(t_peaks) > 1:
                            m = min(len(s_peaks), len(t_peaks))
                            v = np.nanmean(np.diff(s_peaks[:m]) / np.diff(t_peaks[:m]))

                        # Fill current trace metrics.
                        lsts['s'].append(s_val)
                        lsts['t'].append(t_val)
                        lsts['v'].append(v)
                        lsts['rms'].append(rms)
                        lsts['vxpk'].append(vxpk)

        # Only keep the band if at least one finite temporal period exists.
        if np.isfinite(lsts['t']).any():
            L = float(b)
            recs[L] = lsts
            for k in keys:
                pooled[k].extend(lsts[k])

    return recs, pooled


# ---------- stats & utility functions ----------------------------------------

def reject(a: np.ndarray, s: float = 3) -> np.ndarray:
    """
    Robust outlier rejection based on median ± s * MAD.

    Parameters
    ----------
    a : array-like
        Input data.
    s : float
        Threshold in units of MAD (default 3).

    Returns
    -------
    np.ndarray
        Filtered array containing only "inliers".
    """
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return a
    med = np.median(a)
    mad = np.median(np.abs(a - med)) or 1.0
    return a[np.abs(a - med) <= s * mad * 1.4826]


def mean_sd(a: np.ndarray) -> Tuple[float, float]:
    """
    Compute mean and standard deviation (SD) for finite entries.

    Parameters
    ----------
    a : array-like

    Returns
    -------
    mean : float
    sd   : float
        SD is returned as NaN when n < 2.
    """
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.nan, np.nan
    if a.size == 1:
        return float(a[0]), np.nan
    return a.mean(), a.std(ddof=1)


# Units for pretty-printing.
_T = {
    's': 'µm',
    't': 'h',
    'v': 'µm/h',
    'rms': 'µm/h',
    'vxpk': 'µm/h'
}


# ---- temporal periods for standing waves only ------------------------------

def pool_t_for_standing_waves(recs: Dict[float, Dict[str, List[float]]],
                              Lmin: float = 450.0,
                              s_threshold: float = 350.0) -> List[float]:
    """
    Extract temporal periods 't' only for "good" standing waves.

    Criteria:
      * Use only bands with L >= Lmin.
      * Keep traces where both s and t are finite and s > s_threshold.

    Parameters
    ----------
    recs : dict
        Per-band metrics.
    Lmin : float
        Minimum band length to consider.
    s_threshold : float
        Minimum spatial period to be considered a standing wave.

    Returns
    -------
    list of float
        All selected temporal periods pooled together.
    """
    t_vals = []
    for L, lsts in recs.items():
        if L < Lmin:
            continue
        s_arr = np.asarray(lsts['s'], float)
        t_arr = np.asarray(lsts['t'], float)
        mask = np.isfinite(s_arr) & np.isfinite(t_arr) & (s_arr > s_threshold)
        if mask.any():
            t_vals.extend(t_arr[mask].tolist())
    return t_vals


# ---- filter out unwanted bands (e.g., 430 µm) ------------------------------

def filter_recs_exclude(recs, exclude_lengths=(430,)):
    """
    Return a copy of `recs` without bands whose length is in exclude_lengths.
    """
    excl = set(float(x) for x in exclude_lengths)
    return {L: recs[L] for L in sorted(recs) if float(L) not in excl}


# ---- Wilson CI for binomial proportions ------------------------------------

def wilson_ci(k: int, n: int, z: float = 1.959963984540054):
    """
    Compute Wilson confidence interval for a binomial proportion.

    Parameters
    ----------
    k : int
        Number of "successes".
    n : int
        Total number of trials.
    z : float
        z-score for the confidence level (approx. 1.96 for 95% CI).

    Returns
    -------
    center : float
    lower  : float
    upper  : float
        All between 0 and 1. Returns NaNs when n == 0.
    """
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return center, max(0.0, center - half), min(1.0, center + half)


# ---- occurrence plot (no table), excluding 430 by default ------------------

def plot_occurrence_with_wilson(recs, canonical=None, exclude_lengths=(430,)):
    """
    Plot the occurrence of:
      * "nodal oscillations" (multi-peak standing waves),
      * "global oscillations" (non-finite s),
    as a function of band length, with Wilson confidence intervals.
    """
    recs_use = filter_recs_exclude(recs, exclude_lengths=exclude_lengths)
    Ls = sorted(recs_use)
    pm, lom, him = [], [], []
    pg, log, hig = [], [], []

    for L in Ls:
        s = np.asarray(recs_use[L].get('s', []), float)
        t = np.asarray(recs_use[L].get('t', []), float)

        # "Global" oscillations here are traces where s is non-finite.
        mask_global = ~np.isfinite(s)
        # "Nodal" oscillations: finite s and t, with s above threshold.
        mask_multi = np.isfinite(s) & np.isfinite(t) & (s > 350)

        n = len(s)
        km = int(mask_multi.sum())
        kg = int(mask_global.sum())

        cm, lm, hm = wilson_ci(km, n)
        cg, lg, hg = wilson_ci(kg, n)

        pm.append(100 * cm)
        lom.append(100 * lm)
        him.append(100 * hm)
        pg.append(100 * cg)
        log.append(100 * lg)
        hig.append(100 * hg)

    Ls = np.array(Ls)
    pm = np.array(pm); lom = np.array(lom); him = np.array(him)
    pg = np.array(pg); log = np.array(log); hig = np.array(hig)

    # Optional selection of a canonical subset of bands (e.g. those shown in figures).
    if canonical is not None:
        idx = [np.where(Ls == b)[0][0] for b in canonical if b in set(Ls)]
        Ls, pm, lom, him, pg, log, hig = (
            arr[idx] for arr in (Ls, pm, lom, him, pg, log, hig)
        )

    plt.figure(figsize=(4, 3))

    # Light shading for short and long-band regimes (cosmetic).
    if len(Ls) > 0:
        plt.axvspan(0, 390, color='skyblue', alpha=0.2)
        plt.axvspan(1000, 2000, color='pink', alpha=0.15)

    # Confidence bands for nodal and global oscillations.
    plt.fill_between(Ls, lom, him, alpha=0.18, color='r')
    plt.fill_between(Ls, log, hig, alpha=0.18, color='b')

    plt.plot(Ls, pm, 'o-', color='r', label='Nodal oscillations')
    plt.plot(Ls, pg, 'o-', color='b', label='Global oscillations')

    plt.xlabel('Lx (µm)', fontsize=12)
    plt.ylabel('Occurrence (%)', fontsize=12)
    plt.xticks(Ls, [str(int(x)) for x in Ls], rotation=45)
    plt.ylim(-5, 105)

    if len(Ls) > 0:
        plt.xlim(
            min(Ls) - 0.1 * (max(Ls) - min(Ls)),
            max(Ls) + 0.1 * (max(Ls) - min(Ls))
        )

    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()


# ---- table-only figure (counts per band) -----------------------------------

def plot_counts_table_only(
    recs,
    exclude_lengths=(430,),
    fig_w=6.0, fig_h=6.0,
    table_fontsize=12,
    col_widths=(0.34, 0.30, 0.36),
    table_cellpad=0.012,
    save_path=None,
    dpi=600
):
    """
    Draw a publication-quality table listing, for each band length:
      * total number of traces,
      * number of traces with finite temporal period.

    The function can either display the figure interactively or save it to disk.
    """
    import matplotlib as mpl
    # Use TrueType fonts in vector exports so text is editable in Illustrator, etc.
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42

    recs_use = filter_recs_exclude(recs, exclude_lengths=exclude_lengths)
    rows = []
    for L in sorted(recs_use):
        s = np.asarray(recs_use[L]['s'], float)
        t = np.asarray(recs_use[L]['t'], float)
        rows.append((int(L), len(s), int(np.isfinite(t).sum())))

    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_subplot(111)
    ax.axis('off')

    col_labels = [r"$L_x$ (µm)", r"$N_{\mathrm{traces}}$", r"$N_{\mathrm{finite}\,T}$"]
    cell_text = [[str(v) for v in r] for r in rows]

    tbl = ax.table(cellText=cell_text,
                   colLabels=col_labels,
                   cellLoc='center',
                   colLoc='center',
                   loc='center',
                   colWidths=list(col_widths))
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(table_fontsize)
    tbl.scale(1.0, 1.0)

    # Thin borders and reduced padding so the table looks compact.
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_text_props(weight='normal')
        cell.PAD = table_cellpad
        cell.set_linewidth(0.4)

    plt.tight_layout()

    if save_path:
        ext = save_path.lower()
        if ext.endswith(('.pdf', '.svg', '.eps')):
            fig.savefig(save_path, bbox_inches='tight')
        else:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0.02)
        plt.close(fig)
    else:
        fig.set_dpi(200)
        plt.show()


# ---- histograms for pooled metrics -----------------------------------------

def plot_hist(pool):
    """
    Plot simple histograms for each pooled metric.

    The 'settings' dictionary can be tuned to match the aesthetics of a
    specific figure (x-limits, ticks, etc.).
    """
    settings = {
        's':    ('Spatial period', 'µm', (150, 800), [200, 400, 600], [0, 30, 60]),
        't':    ('Temporal period', 'h', (1, 7), [2, 4, 6], [0, 15, 30]),
        'v':    ('Velocity', 'µm/h', None, None, None),
        'rms':  ('RMS', 'µm/h', (10, 35), [15, 20, 25, 30], [0, 100, 200]),
        'vxpk': ('Peak $|V_x|$', 'µm/h', None, None, None)
    }

    for k, arr in pool.items():
        arr = np.asarray(arr, float)
        arr = reject(arr)
        if arr.size == 0:
            continue

        name, unit, xlim, xticks, yticks = settings.get(k, (k, '', None, None, None))

        plt.figure(figsize=(4, 4))
        counts, _bins, _ = plt.hist(arr, bins=9, edgecolor='black', alpha=0.7, color='gray')

        if xlim:
            plt.xlim(*xlim)
        if xticks is not None:
            plt.xticks(xticks, fontsize=15)
        else:
            plt.xticks(fontsize=15)

        if yticks is not None:
            plt.yticks(yticks, fontsize=15)
        else:
            ymax = counts.max()
            if ymax < 10:
                # Discrete ticks when counts are small.
                plt.yticks(np.arange(0, max(6, int(ymax) + 1)), fontsize=15)
                plt.ylim(0, max(5, int(ymax * 1.2)))
            else:
                plt.yticks(fontsize=15)

        if unit:
            plt.xlabel(f"{name} ({unit})", fontsize=20)
        else:
            plt.xlabel(name, fontsize=20)

        plt.ylabel('Frequency', fontsize=20)
        plt.tight_layout()
        plt.show()


# ---- summary metrics as a function of band length --------------------------

def plot_summary_metrics_vs_band(recs):
    """
    Plot band-averaged metrics (mean ± SD) versus band length Lx.
    """
    Lx_vals = [100, 250, 540, 720, 1000, 1500, 2000]
    metrics = [
        ('s', "Spatial period (µm)"),
        ('t', "Temporal period (h)"),
        ('rms', "RMS velocity (µm/h)"),
        ('vxpk', "Peak $|V_x|$ (µm/h)")
    ]

    for k, label in metrics:
        means, sds, Lx_used = [], [], []
        for L in Lx_vals:
            if L not in recs:
                continue
            vals = reject(recs[L].get(k, []))
            m, sd = mean_sd(vals)
            means.append(m)
            sds.append(sd)
            Lx_used.append(L)

        if len(Lx_used) == 0:
            continue

        plt.figure(figsize=(4, 4))
        plt.errorbar(Lx_used, means, yerr=sds, fmt='o-', color='black',
                     capsize=5, elinewidth=2, markeredgewidth=2)
        plt.xticks(Lx_used, fontsize=14, rotation=45)
        plt.yticks(fontsize=14)
        plt.xlabel("Lx (µm)", fontsize=18)
        plt.ylabel(label, fontsize=18)
        plt.tight_layout()
        plt.show()


# ---------- pretty-printing of means / SD -----------------------------------

def print_means_sem(recs,
                    t_Lmin: float = 450.0,
                    t_s_threshold: float = 350.0):
    """
    Print band-wise and global averages of the main metrics.

    All uncertainties printed as "±" are standard deviations (SD), not SEM.

    Temporal period 't' is pooled only over "standing-wave" traces defined
    by L >= t_Lmin and s > t_s_threshold.
    """
    bands = sorted(recs)
    global_stats = {'s': [], 't': [], 'rms': [], 'vxpk': []}

    for k in _T:
        print(f"\n=== {k} ({_T[k]}) ===")
        for L in bands:
            vals = reject(recs[L].get(k, []))
            m, sd = mean_sd(vals)
            print(f"L={L:.0f} : {m:.2f} ± {sd:.2f}")
            if k in global_stats:
                global_stats[k].extend(list(vals))

    print("\n=== GLOBAL AVERAGES (all bands) ===")

    # Global spatial period.
    vals_s = np.array(global_stats['s'], float)
    vals_s = vals_s[np.isfinite(vals_s)]
    m_s, sd_s = mean_sd(vals_s)
    print(f"s ({_T['s']}): {m_s:.2f} ± {sd_s:.2f} (n={len(vals_s)})")

    # Global temporal period for standing waves only.
    vals_t = np.array(
        pool_t_for_standing_waves(recs, Lmin=t_Lmin, s_threshold=t_s_threshold),
        float
    )
    vals_t = vals_t[np.isfinite(vals_t)]
    m_t, sd_t = mean_sd(vals_t)
    print(f"t ({_T['t']}): {m_t:.2f} ± {sd_t:.2f} (n={len(vals_t)})")

    # Global RMS velocity.
    vals_r = np.array(global_stats['rms'], float)
    vals_r = vals_r[np.isfinite(vals_r)]
    m_r, sd_r = mean_sd(vals_r)
    print(f"rms ({_T['rms']}): {m_r:.2f} ± {sd_r:.2f} (n={len(vals_r)})")

    # Global peak |Vx|.
    vals_vxpk = np.array(global_stats['vxpk'], float)
    vals_vxpk = vals_vxpk[np.isfinite(vals_vxpk)]
    m_vxpk, sd_vxpk = mean_sd(vals_vxpk)
    print(f"vxpk ({_T['vxpk']}): {m_vxpk:.2f} ± {sd_vxpk:.2f} (n={len(vals_vxpk)})")


def summary(recs):
    """
    Print a compact summary of band lengths, number of traces, and
    number of traces with finite temporal period.
    """
    print("\nBand\tN\tfinite t")
    for L in sorted(recs):
        t = np.asarray(recs[L]['t'], float)
        print(f"{int(L):>4}\t{len(t)}\t{np.isfinite(t).sum()}")


# ---------- main analysis entry point ----------------------------------------

def analyse(base: str,
            cache_file: str = None,
            use_cache_only: bool = True):
    """
    High-level driver function.

    Parameters
    ----------
    base : str
        Path to the root folder containing band subdirectories.
    cache_file : str or None
        Path to the cache file. If None, 'Sample-data.pkl' in the
        script directory is used.
    use_cache_only : bool
        If True (default), the function *only* loads the cache file and
        never re-runs the .mat analysis. This guarantees bit-for-bit
        reproducibility as long as the cache file stays unchanged.
        If False, the cache will be created or refreshed from the raw
        .mat files when necessary.
    """
    if cache_file is None:
        cache_file = get_cache_path()

    if use_cache_only:
        # Strictly cache-based workflow: the raw .mat files are not touched.
        if not os.path.exists(cache_file):
            raise FileNotFoundError(
                f"Cache file '{cache_file}' not found.\n"
                "Provide a precomputed 'Sample-data.pkl' file in the same "
                "directory as this script, or call analyse(..., "
                "use_cache_only=False) once to generate it from the raw "
                ".mat files."
            )

        print(f"Loading cached data from {cache_file}")
        with open(cache_file, 'rb') as f:
            cache = pickle.load(f)
        recs = cache.get('recs', {})
        pool = cache.get('pool', {})
        cached_ver = cache.get('version', 0)

        # Bring old caches up to date with the current schema if needed.
        if cached_ver < CACHE_VERSION:
            print("Upgrading old cache schema…")
            recs, pool = retrofit_cache(recs, pool)
            with open(cache_file, 'wb') as f:
                pickle.dump({'recs': recs, 'pool': pool, 'version': CACHE_VERSION}, f)
            print("Cache upgraded and resaved.")

    else:
        # Flexible workflow: use cache if available, otherwise recompute from .mat files.
        if os.path.exists(cache_file):
            print(f"Loading cached data from {cache_file}")
            with open(cache_file, 'rb') as f:
                cache = pickle.load(f)
            recs = cache.get('recs', {})
            pool = cache.get('pool', {})
            cached_ver = cache.get('version', 0)

            if cached_ver < CACHE_VERSION:
                print("Upgrading old cache schema…")
                recs, pool = retrofit_cache(recs, pool)
                with open(cache_file, 'wb') as f:
                    pickle.dump({'recs': recs, 'pool': pool, 'version': CACHE_VERSION}, f)
                print("Cache upgraded and resaved.")
        else:
            print("Analyzing .mat files (this may take some time)...")
            recs, pool = collect(base)
            with open(cache_file, 'wb') as f:
                pickle.dump({'recs': recs, 'pool': pool, 'version': CACHE_VERSION}, f)
            print(f"Analysis results cached to {cache_file}")

    # Print a quick summary and then generate all figures/statistics.
    summary(recs)

    # Copy the pooled metrics and enforce "standing-wave only" for temporal periods.
    pool_sw = {k: np.asarray(v) for k, v in pool.items()}
    pool_sw['t'] = np.asarray(pool_t_for_standing_waves(recs, Lmin=450.0, s_threshold=350.0))

    # Histograms (s, t, v, rms, vxpk).
    plot_hist(pool_sw)

    # Per-band and global statistics (mean ± SD).
    print_means_sem(recs)

    # Summary versus band length (mean ± SD as errorbars).
    plot_summary_metrics_vs_band(recs)

    # Occurrence plot for nodal vs global oscillations, excluding Lx = 430 µm.
    canonical_bands = [100, 250, 540, 720, 1000, 1500, 2000]
    plot_occurrence_with_wilson(recs, canonical=canonical_bands, exclude_lengths=(430,))

    # Table-only figure with counts per band.
    plot_counts_table_only(
        recs,
        exclude_lengths=(430,),
        fig_w=6.0,
        fig_h=6.0,
        table_fontsize=12,
        col_widths=(0.34, 0.30, 0.36),
        table_cellpad=0.012,
        save_path=None,
        dpi=600
    )


def main():
    """
    Example entry point.

    Change 'base' to the folder containing your band subdirectories.
    """
    base = r"C:/Users/Genesis/Desktop/PIV-DATA"
    analyse(base)


if __name__ == "__main__":
    main()
