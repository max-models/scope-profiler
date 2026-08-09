import os
import time

from scope_profiler import ProfileManager, read_h5
from scope_profiler.plotting_scripts import plot_flame, plot_gantt

ProfileManager.setup(
    use_likwid=False,
    time_trace=True,
    flush_to_disk=True,
    file_path="profiling_data.h5",
)


@ProfileManager.profile("fibonacci")
def fibonacci(n):
    if n < 2:
        time.sleep(0.001)  # stand-in for leaf work, so bar widths are visible
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


fibonacci(6)
results = ProfileManager.finalize(return_results=True)
results.print_summary()
df = results.to_dataframe()


output_dir = "figures"
os.makedirs(output_dir, exist_ok=True)
flame_path = os.path.join(output_dir, "self_recursive_flame_plot.png")
plot_flame(results, filepath=flame_path, show=False)
plot_gantt(results, show=False)
