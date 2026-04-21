import subprocess
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import os
import threading
import numpy as np

# ── Config ─────────────────────────────────────────────────────────────────────
SEQ_SIZES    = [10000, 100000]
PAR_SIZES    = [10000, 100000, 1000000]
TEST_SIZES   = PAR_SIZES
PREDICT_SIZE = 1_000_000
VERSIONS     = [
    {"name": "Sequential",  "exe": "./sequential_live",  "json": "results.json",        "color": "#FF6B6B", "sizes": SEQ_SIZES},
    {"name": "std::thread", "exe": "./thread_sort",      "json": "results_thread.json", "color": "#4ECDC4", "sizes": PAR_SIZES},
    {"name": "OpenMP",      "exe": "./openmp_sort",      "json": "results_openmp.json", "color": "#FFE66D", "sizes": PAR_SIZES},
    {"name": "CUDA",        "exe": None,                  "json": "results_cuda.json",   "color": "#A855F7", "sizes": PAR_SIZES},
]

# ── Compile ────────────────────────────────────────────────────────────────────
def compile_all():
    jobs = [
        ("sequential_live", ["g++", "-O2", "-std=c++17", "-o", "sequential_live", "sequential_live.cpp"]),
        ("thread_sort",     ["g++", "-O2", "-std=c++17", "-pthread", "-o", "thread_sort", "thread_sort.cpp"]),
        ("openmp_sort",     ["g++", "-O2", "-std=c++17", "-fopenmp", "-o", "openmp_sort", "openmp_sort.cpp"]),
    ]
    for name, cmd in jobs:
        print(f"[*] Compile: {name} ...")
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"    ✗ FAILED:\n{r.stderr}")
            return False
        print(f"    ✓ OK")
    print()
    return True

# ── Read JSON ──────────────────────────────────────────────────────────────────
def read_json(path):
    data = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        e = json.loads(line)
                        data[e["n"]] = e
        except:
            pass
    return data

# ── O(N²) prediction ───────────────────────────────────────────────────────────
def predict_n2(results, target_n):
    ns    = sorted(results.keys())
    times = [results[n]["time_ms"] for n in ns]
    if len(ns) < 1:
        return None
    log_ns = np.log(ns)
    log_ts = np.log(times)
    if len(ns) == 1:
        c = times[0] / (ns[0] ** 2)
        return c * target_n ** 2
    coeffs = np.polyfit(log_ns, log_ts, 1)
    exp, log_c = coeffs
    return np.exp(log_c) * (target_n ** exp)

def fmt_time(ms):
    if ms is None:
        return "N/A"
    if ms < 1000:
        return f"{ms:.0f} ms"
    elif ms < 60000:
        return f"{ms/1000:.1f} sec"
    elif ms < 3600000:
        return f"{ms/60000:.1f} min"
    else:
        return f"{ms/3600000:.2f} hr"

# ── State ──────────────────────────────────────────────────────────────────────
all_results  = [{} for _ in VERSIONS]
current_ver  = [0]
run_done     = [False]

# ── Runner thread ──────────────────────────────────────────────────────────────
def runner():
    for i, ver in enumerate(VERSIONS):
        current_ver[0] = i
        if ver["exe"] is None:
            continue  # CUDA: pre-computed JSON, skip running
        if os.path.exists(ver["json"]):
            os.remove(ver["json"])
        proc = subprocess.Popen(
            [ver["exe"]], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        proc.wait()
    run_done[0] = True

# ── Matplotlib setup ───────────────────────────────────────────────────────────
plt.style.use("dark_background")
fig = plt.figure(figsize=(22, 11))
fig.patch.set_facecolor("#0a0a0a")

gs = gridspec.GridSpec(3, 4, figure=fig,
                       hspace=0.55, wspace=0.38,
                       left=0.05, right=0.97, top=0.91, bottom=0.07)

ax_prog  = [fig.add_subplot(gs[0, c]) for c in range(4)]
ax_table = fig.add_subplot(gs[1, :])
ax_time  = fig.add_subplot(gs[2, 0])
ax_pred  = fig.add_subplot(gs[2, 1])
ax_spdup = fig.add_subplot(gs[2, 2:])

# ── Draw: progress bars ────────────────────────────────────────────────────────
def draw_progress(ax, ver_idx, results):
    ver = VERSIONS[ver_idx]
    sizes = ver["sizes"]
    ax.clear()
    ax.set_facecolor("#111111")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(sizes) + 1.4)
    ax.axis("off")
    is_active = (ver_idx == current_ver[0] and not run_done[0])
    ax.set_title(ver["name"], color=ver["color"] if is_active else "#888",
                 fontsize=11, fontweight="bold", pad=6)

    for i, n in enumerate(sizes):
        y = len(sizes) - i - 0.5
        ax.add_patch(Rectangle((0.1, y - 0.33), 9.8, 0.66,
                               facecolor="#1a1a1a", edgecolor="#333", lw=0.8))
        label = f"N={n:>9,}"
        if ver_idx == 0 and n == PREDICT_SIZE:
            p = predict_n2(results, PREDICT_SIZE)
            ax.text(0.0, y, label, ha="left", va="center", color="#FFA500", fontsize=8)
            if p:
                ax.add_patch(Rectangle((0.1, y - 0.33), 9.8, 0.66,
                                       facecolor="#553300", edgecolor="#FFA500", lw=1, alpha=0.7))
                ax.text(5.0, y, f"~{fmt_time(p)}  (predicted)", ha="center", va="center",
                        color="#FFA500", fontsize=8, fontweight="bold")
        else:
            ax.text(0.0, y, label, ha="left", va="center", color="#666", fontsize=8)
            if n in results:
                e = results[n]
                ax.add_patch(Rectangle((0.1, y - 0.33), 9.8, 0.66,
                                       facecolor=ver["color"], edgecolor="white", lw=0.5, alpha=0.8))
                ax.text(5.0, y, fmt_time(e["time_ms"]), ha="center", va="center",
                        color="#000", fontsize=9, fontweight="bold")
            elif is_active and ver["exe"] is not None:
                ax.text(5.0, y, "running...", ha="center", va="center",
                        color="#555", fontsize=9, style="italic")

# ── Draw: results + prediction table ──────────────────────────────────────────
def draw_table(ax, all_res):
    ax.clear()
    ax.set_facecolor("#111111")
    ax.axis("off")
    rows = len(TEST_SIZES) + 1
    ax.set_xlim(0, 10)
    ax.set_ylim(0, rows + 2)

    headers = ["N", "Sequential", "std::thread", "OpenMP", "CUDA", "Spdup(Thr)", "Spdup(OMP)", "Spdup(CUDA)"]
    xs      = [0.1,  1.3,          2.6,           3.9,      5.2,    6.35,         7.5,           8.65]
    for x, h in zip(xs, headers):
        ax.text(x, rows + 1.5, h, ha="left", va="center",
                color="#ccccff", fontsize=8, fontweight="bold")
    ax.plot([0.1, 9.9], [rows + 1.0, rows + 1.0], color="#444", lw=0.8)

    for i, n in enumerate(TEST_SIZES):
        y = rows - i - 0.5
        seq  = all_res[0].get(n, {})
        thr  = all_res[1].get(n, {})
        omp  = all_res[2].get(n, {})
        cuda = all_res[3].get(n, {})

        ax.text(xs[0], y, f"{n:,}", ha="left", va="center", color="#aaa", fontsize=8)
        ax.text(xs[1], y, fmt_time(seq.get("time_ms"))  if seq  else "—", ha="left", va="center", color=VERSIONS[0]["color"], fontsize=8)
        ax.text(xs[2], y, fmt_time(thr.get("time_ms"))  if thr  else "—", ha="left", va="center", color=VERSIONS[1]["color"], fontsize=8)
        ax.text(xs[3], y, fmt_time(omp.get("time_ms"))  if omp  else "—", ha="left", va="center", color=VERSIONS[2]["color"], fontsize=8)
        ax.text(xs[4], y, fmt_time(cuda.get("time_ms")) if cuda else "—", ha="left", va="center", color=VERSIONS[3]["color"], fontsize=8)

        for col, (a, b) in zip([xs[5], xs[6], xs[7]], [(seq, thr), (seq, omp), (seq, cuda)]):
            if a and b:
                sp = a["time_ms"] / b["time_ms"]
                ax.text(col, y, f"×{sp:.2f}", ha="left", va="center",
                        color="#00FF87" if sp >= 1 else "#FF6B6B", fontsize=8, fontweight="bold")
            else:
                ax.text(col, y, "—", ha="left", va="center", color="#444", fontsize=8)

    ax.plot([0.1, 9.9], [rows - len(TEST_SIZES) - 0.05, rows - len(TEST_SIZES) - 0.05],
            color="#444", lw=0.6, linestyle="--")

    pred_y = 0.5
    ax.text(xs[0], pred_y, f"1,000,000", ha="left", va="center",
            color="#FFA500", fontsize=8, fontweight="bold")

    p_seq = predict_n2(all_res[0], PREDICT_SIZE) if all_res[0] else None
    ax.text(xs[1], pred_y, f"~{fmt_time(p_seq)} (est.)" if p_seq else "—",
            ha="left", va="center", color="#FF6B6B", fontsize=7, fontstyle="italic")

    for vi, col in zip([1, 2, 3], [xs[2], xs[3], xs[4]]):
        actual = all_res[vi].get(PREDICT_SIZE)
        if actual:
            ax.text(col, pred_y, fmt_time(actual["time_ms"]), ha="left", va="center",
                    color=VERSIONS[vi]["color"], fontsize=8, fontweight="bold")
        else:
            p = predict_n2(all_res[vi], PREDICT_SIZE) if all_res[vi] else None
            label = f"~{fmt_time(p)} (est.)" if p else "running..."
            ax.text(col, pred_y, label, ha="left", va="center",
                    color=VERSIONS[vi]["color"], fontsize=7, fontstyle="italic")

    p_thr_val  = all_res[1].get(PREDICT_SIZE, {}).get("time_ms") or predict_n2(all_res[1], PREDICT_SIZE)
    p_omp_val  = all_res[2].get(PREDICT_SIZE, {}).get("time_ms") or predict_n2(all_res[2], PREDICT_SIZE)
    p_cuda_val = all_res[3].get(PREDICT_SIZE, {}).get("time_ms") or predict_n2(all_res[3], PREDICT_SIZE)
    for col, p_par in zip([xs[5], xs[6], xs[7]], [p_thr_val, p_omp_val, p_cuda_val]):
        if p_seq and p_par:
            sp = p_seq / p_par
            ax.text(col, pred_y, f"×{sp:.1f}", ha="left", va="center",
                    color="#00FF87", fontsize=8, fontweight="bold")
        else:
            ax.text(col, pred_y, "—", ha="left", va="center", color="#444", fontsize=8)

# ── Draw: actual time bar chart ────────────────────────────────────────────────
def draw_time_chart(ax, all_res):
    ax.clear()
    ax.set_facecolor("#1a1a1a")
    common = [n for n in TEST_SIZES if any(n in r for r in all_res)]
    if not common:
        ax.text(0.5, 0.5, "Waiting...", ha="center", va="center",
                color="#555", transform=ax.transAxes)
        return
    x = np.arange(len(common))
    n_ver = len(VERSIONS)
    w = 0.8 / n_ver
    for vi, ver in enumerate(VERSIONS):
        times = [all_res[vi][n]["time_ms"] if n in all_res[vi] else 0 for n in common]
        offset = (vi - (n_ver - 1) / 2) * w
        ax.bar(x + offset, times, width=w,
               color=ver["color"], alpha=0.85, edgecolor="white", lw=0.4,
               label=ver["name"])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n:,}" for n in common], color="white", fontsize=8)
    ax.set_ylabel("Time (ms)", color="white", fontsize=9)
    ax.set_title("Actual Time", color="white", fontsize=10, fontweight="bold")
    ax.tick_params(colors="white")
    ax.grid(axis="y", linestyle=":", alpha=0.2)
    ax.legend(fontsize=7, facecolor="#111", edgecolor="#444", labelcolor="white")

# ── Draw: prediction curve ─────────────────────────────────────────────────────
def draw_prediction(ax, all_res):
    ax.clear()
    ax.set_facecolor("#1a1a1a")
    has_data = any(len(r) >= 1 for r in all_res)
    if not has_data:
        ax.text(0.5, 0.5, "Waiting...", ha="center", va="center",
                color="#555", transform=ax.transAxes)
        ax.set_title("N=1M Prediction (O(N²))", color="white", fontsize=10, fontweight="bold")
        return

    ns_plot = np.logspace(np.log10(8000), np.log10(1_200_000), 200)
    for vi, ver in enumerate(VERSIONS):
        res = all_res[vi]
        if not res:
            continue
        ns_known = sorted(res.keys())
        ts_known = [res[n]["time_ms"] for n in ns_known]
        log_ns = np.log(ns_known)
        log_ts = np.log(ts_known)
        if len(ns_known) == 1:
            c = ts_known[0] / (ns_known[0] ** 2)
            ts_curve = c * ns_plot ** 2
        else:
            coeffs = np.polyfit(log_ns, log_ts, 1)
            ts_curve = np.exp(coeffs[1]) * ns_plot ** coeffs[0]
        ax.plot(ns_plot, ts_curve, color=ver["color"], lw=2, alpha=0.85, label=ver["name"])
        ax.scatter(ns_known, ts_known, color=ver["color"], s=50, zorder=5)

    ax.axvline(PREDICT_SIZE, color="#FFA500", linestyle="--", lw=1.5, alpha=0.7, label="N=1M")
    for vi, ver in enumerate(VERSIONS):
        p = predict_n2(all_res[vi], PREDICT_SIZE) if all_res[vi] else None
        if p:
            ax.scatter([PREDICT_SIZE], [p], color=ver["color"], s=80, marker="*", zorder=6)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N (elements)", color="white", fontsize=9)
    ax.set_ylabel("Time (ms, log)", color="white", fontsize=9)
    ax.set_title("N=1M Prediction  (fit)", color="white", fontsize=10, fontweight="bold")
    ax.tick_params(colors="white")
    ax.grid(linestyle=":", alpha=0.2)
    ax.legend(fontsize=7, facecolor="#111", edgecolor="#444", labelcolor="white")

# ── Draw: speedup bars ─────────────────────────────────────────────────────────
def draw_speedup(ax, all_res):
    ax.clear()
    ax.set_facecolor("#1a1a1a")
    par_versions = [1, 2, 3]  # thread, openmp, cuda
    common = [n for n in TEST_SIZES
              if all_res[0].get(n) and any(all_res[vi].get(n) for vi in par_versions)]
    if not common:
        ax.text(0.5, 0.5, "Waiting...", ha="center", va="center",
                color="#555", transform=ax.transAxes)
        ax.set_title("Speedup vs Sequential", color="white", fontsize=10, fontweight="bold")
        return

    x = np.arange(len(common))
    labels = [f"{n:,}" for n in common]
    n_par = len(par_versions)
    w = 0.75 / n_par

    for j, vi in enumerate(par_versions):
        speedups = []
        for n in common:
            seq_t = all_res[0][n]["time_ms"] if all_res[0].get(n) else None
            par_t = all_res[vi][n]["time_ms"] if all_res[vi].get(n) else None
            speedups.append(seq_t / par_t if seq_t and par_t else 0)
        offset = (j - (n_par - 1) / 2) * w
        ax.bar(x + offset, speedups, w, color=VERSIONS[vi]["color"], alpha=0.85,
               label=VERSIONS[vi]["name"], edgecolor="white", lw=0.4)

    ax.axhline(1.0, color="#FF6B6B", linestyle="--", lw=1.2, alpha=0.7, label="Baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color="white", fontsize=8)
    ax.set_ylabel("Speedup (×)", color="white", fontsize=9)
    ax.set_title("Speedup vs Sequential", color="white", fontsize=10, fontweight="bold")
    ax.tick_params(colors="white")
    ax.grid(axis="y", linestyle=":", alpha=0.2)
    ax.legend(fontsize=7, facecolor="#111", edgecolor="#444", labelcolor="white")

# ── Animation update ───────────────────────────────────────────────────────────
def update(frame):
    for i, ver in enumerate(VERSIONS):
        all_results[i] = read_json(ver["json"])
    for i in range(4):
        draw_progress(ax_prog[i], i, all_results[i])
    draw_table(ax_table, all_results)
    draw_time_chart(ax_time, all_results)
    draw_prediction(ax_pred, all_results)
    draw_speedup(ax_spdup, all_results)

    done  = sum(len(r) for r in all_results)
    total = sum(len(ver["sizes"]) for ver in VERSIONS)
    cur   = VERSIONS[current_ver[0]]["name"] if not run_done[0] else "DONE!"
    fig.suptitle(
        f"Parallel Sort Benchmark  ·  {done}/{total} tests  ·  {cur}",
        color="white", fontsize=13, fontweight="bold", y=0.975
    )

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not compile_all():
        exit(1)

    for ver in VERSIONS:
        if ver["exe"] is not None and os.path.exists(ver["json"]):
            os.remove(ver["json"])

    print("[*] Running: Sequential → std::thread → OpenMP  (CUDA: pre-loaded)\n")
    t = threading.Thread(target=runner, daemon=True)
    t.start()

    ani = animation.FuncAnimation(fig, update, interval=600, cache_frame_data=False)
    plt.show()

    t.join()
    print("\n[✓] Benchmark complete!")
