"""
Plotting with maxplotlib - Export to matplotlib or plotly
==========================================================

This example demonstrates how to use the refactored plotting functions
with maxplotlib, which allows exporting to both matplotlib and plotly backends.

The plotting functions now support a `backend` parameter that can be set to:
- "matplotlib" (default) - exports as PNG/PDF/etc
- "plotly" - exports as interactive HTML

Run this example after profiling some code:

    python examples/ex_maxplotlib_plotting.py
"""

import tempfile
from pathlib import Path

from scope_profiler import ProfileManager
from scope_profiler.h5reader import ProfilingH5Reader
from scope_profiler.plotting_scripts import (
    plot_durations,
    plot_flame,
    plot_gantt,
    plot_speedup,
)


def create_sample_profile(filepath: str):
    """Create a sample profile for demonstration."""
    ProfileManager.setup(
        use_likwid=False,
        time_trace=True,
        flush_to_disk=True,
        file_path=filepath,
    )

    with ProfileManager.profile_region("main"):
        import time

        time.sleep(0.01)

        with ProfileManager.profile_region("compute"):
            time.sleep(0.005)

        with ProfileManager.profile_region("io"):
            time.sleep(0.008)

            with ProfileManager.profile_region("write"):
                time.sleep(0.003)

    ProfileManager.finalize()


def main():
    """Demonstrate maxplotlib plotting with both matplotlib and plotly backends."""
    print("=" * 80)
    print("Maxplotlib Plotting Examples")
    print("=" * 80)

    # Create a temporary profile
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        profile_path = f.name

    print(f"\n1. Creating sample profile: {profile_path}")
    create_sample_profile(profile_path)

    # Load the profile
    reader = ProfilingH5Reader(profile_path)

    # Example 1: Gantt chart with matplotlib
    print("\n2. Creating Gantt chart (matplotlib backend)...")
    gantt_matplotlib = "gantt_matplotlib.png"
    plot_gantt(reader, filepath=gantt_matplotlib, backend="matplotlib", verbose=False)
    print(f"   ✓ Saved to: {gantt_matplotlib}")

    # Example 2: Gantt chart with plotly (note: may have issues with some color formats)
    print("\n3. Creating Gantt chart (plotly backend)...")
    try:
        gantt_plotly = "gantt_plotly.html"
        plot_gantt(reader, filepath=gantt_plotly, backend="plotly", verbose=False)
        print(f"   ✓ Saved to: {gantt_plotly}")
        print(f"   Open {gantt_plotly} in a browser for interactive visualization!")
    except Exception as e:
        print(f"   ⚠ Plotly backend had an issue: {e}")
        print("   This is a known upstream issue with maxplotlib gantt charts")

    # Example 3: Flame chart with matplotlib
    print("\n4. Creating Flame chart (matplotlib backend)...")
    flame_matplotlib = "flame_matplotlib.png"
    plot_flame(reader, filepath=flame_matplotlib, backend="matplotlib", verbose=False)
    print(f"   ✓ Saved to: {flame_matplotlib}")

    # Example 4: Flame chart with plotly
    print("\n5. Creating Flame chart (plotly backend)...")
    flame_plotly = "flame_plotly.html"
    plot_flame(reader, filepath=flame_plotly, backend="plotly", verbose=False)
    print(f"   ✓ Saved to: {flame_plotly}")
    print(f"   Open {flame_plotly} in a browser for interactive visualization!")

    # Example 5: Duration comparison (requires multiple profiles)
    print("\n6. Creating duration comparison chart...")
    # Create a second profile for comparison
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        profile_path2 = f.name
    create_sample_profile(profile_path2)
    reader2 = ProfilingH5Reader(profile_path2)

    durations_matplotlib = "durations_matplotlib.png"
    plot_durations(
        [reader, reader2],
        labels=["Run 1", "Run 2"],
        metrics="avg",
        filepath=durations_matplotlib,
        backend="matplotlib",
        verbose=False,
    )
    print(f"   ✓ Matplotlib: {durations_matplotlib}")

    durations_plotly = "durations_plotly.html"
    plot_durations(
        [reader, reader2],
        labels=["Run 1", "Run 2"],
        metrics="avg",
        filepath=durations_plotly,
        backend="plotly",
        verbose=False,
    )
    print(f"   ✓ Plotly: {durations_plotly}")

    # Example 6: Speedup plot
    print("\n7. Creating speedup plot...")
    speedup_matplotlib = "speedup_matplotlib.png"
    plot_speedup(
        [reader, reader2],
        x_field="num_ranks",
        filepath=speedup_matplotlib,
        backend="matplotlib",
        verbose=False,
    )
    print(f"   ✓ Matplotlib: {speedup_matplotlib}")

    speedup_plotly = "speedup_plotly.html"
    plot_speedup(
        [reader, reader2],
        x_field="num_ranks",
        filepath=speedup_plotly,
        backend="plotly",
        verbose=False,
    )
    print(f"   ✓ Plotly: {speedup_plotly}")

    # Example 7: Export data to JSON for custom visualization
    print("\n8. Exporting plot data to JSON...")
    gantt_data = "gantt_data.json"
    plot_gantt(reader, data_filepath=gantt_data, data_format="json", verbose=False)
    print(f"   ✓ Gantt data: {gantt_data}")

    flame_data = "flame_data.json"
    plot_flame(reader, data_filepath=flame_data, data_format="json", verbose=False)
    print(f"   ✓ Flame data: {flame_data}")

    # Cleanup
    Path(profile_path).unlink()
    Path(profile_path2).unlink()

    print("\n" + "=" * 80)
    print("Summary:")
    print("=" * 80)
    print("✓ All plotting functions now use maxplotlib")
    print("✓ Export to matplotlib (PNG, PDF, etc.) or plotly (interactive HTML)")
    print("✓ Export raw data to JSON/CSV for custom visualizations")
    print("\nKey features:")
    print("  - Unified API across all plot types")
    print("  - Backend selection via 'backend' parameter")
    print("  - Data export for reproducibility")
    print("  - Interactive plotly charts for exploration")
    print("=" * 80)


if __name__ == "__main__":
    main()
