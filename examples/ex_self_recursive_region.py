import os
import time

from scope_profiler import ProfileManager
from scope_profiler.h5reader import ProfilingH5Reader
from scope_profiler.plotting_scripts import plot_flame

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
ProfileManager.finalize()

output_dir = "figures"
os.makedirs(output_dir, exist_ok=True)
flame_path = os.path.join(output_dir, "self_recursive_flame_plot.png")
plot_flame(ProfilingH5Reader("profiling_data.h5"), filepath=flame_path)
print(f"Flame graph written to {flame_path}")
