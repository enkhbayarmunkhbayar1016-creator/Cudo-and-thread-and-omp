import subprocess
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.patches import Rectangle
import os

CPP_FILE = "sequential_live.cpp"
EXE_FILE = "sequential_live"

TEST_SIZES = [1000, 10000, 50000, 100000, 1000000]
MAX_COMPARISONS = 1_000_000_000  # 1 billion

# Compile
def compile_cpp():
    print("[*] Compile хийж байна...")
    result = subprocess.run(
        ["g++", "-O2", "-std=c++17", "-o", EXE_FILE, CPP_FILE],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("Compile алдаа:\n", result.stderr)
        return False
    print("    ✓ OK\n")
    return True

# ══════════════════════════════════════════════════════════════════════════════
# LIVE ANIMATION
# ══════════════════════════════════════════════════════════════════════════════

plt.style.use('dark_background')
fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor('#0a0a0a')

gs = fig.add_gridspec(3, 1, hspace=0.4, left=0.08, right=0.95, top=0.92, bottom=0.08)
ax_live = fig.add_subplot(gs[0, 0])     # Live progress
ax_table = fig.add_subplot(gs[1, 0])    # Results table
ax_chart = fig.add_subplot(gs[2, 0])    # Chart

results_data = {}
process = None
test_order = []

def read_results_json():
    """Read results.json and return dict"""
    data = {}
    if os.path.exists("results.json"):
        try:
            with open("results.json", "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entry = json.loads(line)
                        data[entry["n"]] = entry
        except:
            pass
    return data

def draw_live_progress(ax, results):
    """Draw live progress bar"""
    ax.clear()
    ax.set_facecolor('#111111')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(TEST_SIZES) + 1)
    ax.axis('off')
    ax.set_title("Live Benchmark Progress", color='white', fontsize=13, fontweight='bold', pad=10)

    for i, size in enumerate(TEST_SIZES):
        y = len(TEST_SIZES) - i - 0.5

        # Background bar
        ax.add_patch(Rectangle((0.2, y - 0.3), 9.6, 0.6, 
                               facecolor='#1a1a1a', edgecolor='#444', lw=1))

        if size in results:
            entry = results[size]
            comps = entry["comparisons"]
            stopped = entry.get("stopped_early", False)
            
            # Progress ratio
            ratio = min(1.0, comps / MAX_COMPARISONS)
            
            # Color: green if completed, yellow if stopped
            if stopped:
                color = '#FFA500'  # Orange
                status = "STOPPED (>1M)"
            else:
                color = '#00FF87'  # Green
                status = "DONE"
            
            # Progress bar
            ax.add_patch(Rectangle((0.2, y - 0.3), ratio * 9.6, 0.6,
                                   facecolor=color, edgecolor='white', lw=0.5, alpha=0.8))
            
            # Text
            comps_str = f"{comps:,.0f}" if comps < 1_000_000_000 else f"{comps/1e9:.2f}B"
            ax.text(0.1, y, f"N={size:>7}", ha='right', va='center', 
                   color='#aaa', fontsize=10, fontweight='bold')
            ax.text(5.0, y, comps_str, ha='center', va='center',
                   color='white', fontsize=9, fontweight='bold')
            ax.text(9.8, y, status, ha='right', va='center',
                   color=color, fontsize=9, fontweight='bold')
        else:
            # Waiting
            ax.text(0.1, y, f"N={size:>7}", ha='right', va='center',
                   color='#555', fontsize=10)
            ax.text(5.0, y, "waiting...", ha='center', va='center',
                   color='#555', fontsize=9, style='italic')

def draw_table(ax, results):
    """Draw results table"""
    ax.clear()
    ax.set_facecolor('#1a1a1a')
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(TEST_SIZES) + 2)

    # Header
    ax.text(1.0, len(TEST_SIZES) + 1.5, "N", ha='left', va='center',
           color='#ccccff', fontsize=11, fontweight='bold')
    ax.text(2.5, len(TEST_SIZES) + 1.5, "Comparisons", ha='left', va='center',
           color='#ccccff', fontsize=11, fontweight='bold')
    ax.text(5.5, len(TEST_SIZES) + 1.5, "Time (ms)", ha='left', va='center',
           color='#ccccff', fontsize=11, fontweight='bold')
    ax.text(7.5, len(TEST_SIZES) + 1.5, "Sorted", ha='left', va='center',
           color='#ccccff', fontsize=11, fontweight='bold')
    ax.text(9.0, len(TEST_SIZES) + 1.5, "Status", ha='left', va='center',
           color='#ccccff', fontsize=11, fontweight='bold')

    # Draw line
    ax.plot([0.5, 9.5], [len(TEST_SIZES) + 1.0, len(TEST_SIZES) + 1.0],
           color='#555', lw=1)

    for i, size in enumerate(TEST_SIZES):
        y = len(TEST_SIZES) - i - 0.5

        if size in results:
            entry = results[size]
            comps = entry["comparisons"]
            time_ms = entry["time_ms"]
            sorted_val = "Yes" if entry["sorted"] else "No"
            stopped = entry.get("stopped_early", False)
            status = "STOPPED" if stopped else "✓"
            status_color = '#FFA500' if stopped else '#00FF87'

            comps_str = f"{comps:,.0f}" if comps < 1_000_000_000 else f"{comps/1e9:.2f}B"
            
            ax.text(1.0, y, f"{size:>10,}", ha='left', va='center',
                   color='#aaa', fontsize=9)
            ax.text(2.5, y, comps_str, ha='left', va='center',
                   color='#ccccff', fontsize=9)
            ax.text(5.5, y, f"{time_ms:>8.2f}", ha='left', va='center',
                   color='#ffff99', fontsize=9)
            ax.text(7.5, y, sorted_val, ha='left', va='center',
                   color='#99ff99' if sorted_val == "Yes" else '#ff9999', fontsize=9)
            ax.text(9.0, y, status, ha='left', va='center',
                   color=status_color, fontsize=9, fontweight='bold')

def draw_chart(ax, results):
    """Draw comparisons chart"""
    ax.clear()
    ax.set_facecolor('#1a1a1a')

    if not results:
        ax.text(0.5, 0.5, "Waiting for results...",
               ha='center', va='center', color='#555',
               transform=ax.transAxes, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    ns = sorted([n for n in results.keys()])
    comps = [results[n]["comparisons"] for n in ns]
    colors = ['#FFA500' if results[n].get("stopped_early") else '#00FF87' for n in ns]
    labels = [f"{n:,}" for n in ns]

    bars = ax.bar(labels, comps, color=colors, edgecolor='white', linewidth=0.5, alpha=0.8)

    for bar, c in zip(bars, comps):
        comps_str = f"{c/1e9:.2f}B" if c >= 1_000_000_000 else f"{c/1e6:.1f}M"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(comps)*0.02,
               comps_str, ha='center', va='bottom', color='white', fontsize=9, fontweight='bold')

    # 1M line
    ax.axhline(y=1_000_000_000, color='#FF5555', linestyle='--', linewidth=2, label='1B (1M stop)', alpha=0.7)

    ax.set_xlabel("N (элементийн тоо)", color='white', fontsize=11)
    ax.set_ylabel("Comparisons", color='white', fontsize=11)
    ax.set_title("Element Comparisons per Test Size", color='white', fontsize=12, fontweight='bold')
    ax.tick_params(colors='white')
    ax.grid(axis='y', linestyle=':', alpha=0.25)
    ax.legend(loc='upper left', facecolor='#111', edgecolor='#555')

    # Set y-axis to log scale if big numbers
    if max(comps) > 100_000_000:
        ax.set_yscale('log')
        ax.set_ylabel("Comparisons (log scale)", color='white', fontsize=11)

def update(frame):
    """Animation update"""
    global results_data
    
    # Poll results.json
    results_data = read_results_json()
    
    draw_live_progress(ax_live, results_data)
    draw_table(ax_table, results_data)
    draw_chart(ax_chart, results_data)

    # Title
    completed = len(results_data)
    total = len(TEST_SIZES)
    fig.suptitle(
        f"Sequential Bubble Sort Benchmark  |  {completed}/{total} tests completed  |  "
        f"Max: {max([r['comparisons'] for r in results_data.values()], default=0)/1e9:.2f}B comparisons",
        color='white', fontsize=13, fontweight='bold', y=0.97
    )

# Start C++ process
if not compile_cpp():
    exit(1)

print("[*] Benchmark эхэлж байна...\n")
process = subprocess.Popen([f"./{EXE_FILE}"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# Animation
ani = animation.FuncAnimation(fig, update, interval=500, cache_frame_data=False)

print("Animation дэлгэцэнд гарч байна. Benchmark дуусмагц график үзүүлнэ.\n")
plt.show()

# Wait for process
process.wait()
print("\n[✓] Benchmark дууслаа! results.json хадгалагдлаа.")

# Save final chart
plt.savefig("benchmark_live.png", dpi=120, bbox_inches='tight', facecolor='#0a0a0a')
print("[✓] benchmark_live.png хадгалагдлаа")
