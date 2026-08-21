import os
import time

from scope_profiler import ProfileManager
from scope_profiler.plotting_scripts import plot_flame, plot_gantt


@ProfileManager.profile("fibonacci")
def fibonacci(n):
    if n < 2:
        time.sleep(0.001)  # stand-in for leaf work, so bar widths are visible
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


with ProfileManager.session(
    use_likwid=False, file_path="profiling_data.h5", return_results=True
) as run:
    fibonacci(6)

results = run.results
results.print_summary()
df = results.to_dataframe()


output_dir = "figures"
os.makedirs(output_dir, exist_ok=True)
flame_path = os.path.join(output_dir, "self_recursive_flame_plot.png")
plot_flame(results, filepath=flame_path, show=False)
plot_gantt(results, show=False)
