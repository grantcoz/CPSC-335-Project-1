import random
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, RadioButtons, Slider

# ============================================================
# Sorting Algorithms (Generator Versions)
# ============================================================

def Bubblesort(a):
    n = len(a)
    for i in range(n):
        for j in range(0, n - i - 1):
            if a[j] > a[j + 1]:
                a[j], a[j + 1] = a[j + 1], a[j]
            yield a.copy()
    yield a.copy()


def MergeSort(a):
    def merge_sort(arr, left, right):
        if left < right:
            mid = (left + right) // 2
            yield from merge_sort(arr, left, mid)
            yield from merge_sort(arr, mid + 1, right)
            yield from merge(arr, left, mid, right)

    def merge(arr, left, mid, right):
        L = arr[left:mid + 1]
        R = arr[mid + 1:right + 1]
        i = j = 0
        k = left

        while i < len(L) and j < len(R):
            if L[i] <= R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1
            yield arr.copy()

        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1
            yield arr.copy()

        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1
            yield arr.copy()

    yield from merge_sort(a, 0, len(a) - 1)


def QuickSort(a, low, high):
    def partition(arr, low, high):
        pivot = arr[high]
        i = low - 1
        for j in range(low, high):
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
                yield arr.copy()
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        yield arr.copy()
        return i + 1

    if low < high:
        gen = partition(a, low, high)
        pivot_index = None
        try:
            while True:
                yield next(gen)
        except StopIteration as e:
            pivot_index = e.value

        yield from QuickSort(a, low, pivot_index - 1)
        yield from QuickSort(a, pivot_index + 1, high)


def RadixSort(a):
    def counting(exp):
        n = len(a)
        output = [0] * n
        count = [0] * 10

        for i in range(n):
            index = (a[i] // exp) % 10
            count[index] += 1

        for i in range(1, 10):
            count[i] += count[i - 1]

        for i in range(n - 1, -1, -1):
            digit = (a[i] // exp) % 10
            output[count[digit] - 1] = a[i]
            count[digit] -= 1

        for i in range(n):
            a[i] = output[i]
            yield a.copy()

    max_num = max(a)
    exp = 1
    while max_num // exp > 0:
        yield from counting(exp)
        exp *= 10
    yield a.copy()


# ============================================================
# Algorithm Mapping
# ============================================================

ALGO_BUILDERS = {
    "Bubble": lambda data: Bubblesort(data),
    "Merge":  lambda data: MergeSort(data),
    "Quick":  lambda data: QuickSort(data, 0, len(data) - 1),
    "Radix":  lambda data: RadixSort(data),
}

# ============================================================
# Settings
# ============================================================

LOW, HIGH = 1, 100
SCIENTIFIC_TRIALS = 5
N = 35

data = [random.randint(LOW, HIGH) for _ in range(N)]
original_data = data.copy()

selected_algo = "Bubble"
anim = None
is_running = False

# ============================================================
# Benchmarking
# ============================================================

def benchmark_algorithm(algo_name, dataset):
    test_data = dataset.copy()
    gen = ALGO_BUILDERS[algo_name](test_data)
    start = time.perf_counter()
    for _ in gen:
        pass
    end = time.perf_counter()
    return end - start


def benchmark_scientific(dataset, trials):
    results = {}
    for name in ALGO_BUILDERS:
        total = 0
        for _ in range(trials):
            total += benchmark_algorithm(name, dataset)
        results[name] = total / trials
    return results


# ============================================================
# UI Setup
# ============================================================

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.30, right=0.98, bottom=0.18)

bars = ax.bar(range(N), data)
ax.set_ylim(0, HIGH * 1.1)
ax.set_title("Sorting Visualizer")

# Slider for number of elements
ax_slider = plt.axes([0.30, 0.05, 0.60, 0.04])
slider = Slider(ax_slider, "Number of Elements", 5, 100, valinit=N, valstep=5)

rax = plt.axes([0.05, 0.55, 0.22, 0.35])
radio = RadioButtons(rax, list(ALGO_BUILDERS.keys()))
radio.set_active(0)

ax_start = plt.axes([0.05, 0.45, 0.22, 0.07])
btn_start = Button(ax_start, "Start")

ax_reset = plt.axes([0.05, 0.36, 0.22, 0.07])
btn_reset = Button(ax_reset, "Reset")

ax_new = plt.axes([0.05, 0.27, 0.22, 0.07])
btn_new = Button(ax_new, "New Data")

ax_compare = plt.axes([0.05, 0.18, 0.22, 0.07])
btn_compare = Button(ax_compare, "Compare Algorithms")

status_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")

# ============================================================
# Helper Functions
# ============================================================

def stop_animation():
    global anim, is_running
    if anim is not None:
        if anim.event_source is not None:
            stop_method = getattr(anim.event_source, "stop", None)
            if callable(stop_method):
                stop_method()
        anim = None
    is_running = False


def regenerate_data(new_n):
    global data, original_data, bars
    data = [random.randint(LOW, HIGH) for _ in range(new_n)]
    original_data = data.copy()
    ax.clear()
    ax.set_ylim(0, HIGH * 1.1)
    ax.set_title("Sorting Visualizer")
    bars = ax.bar(range(new_n), data)
    fig.canvas.draw_idle()


def on_slider_change(val):
    stop_animation()
    new_n = int(slider.val)
    regenerate_data(new_n)


slider.on_changed(on_slider_change)


def draw_array(arr):
    for bar, val in zip(bars, arr):
        bar.set_height(val)
    return bars


def on_select(label):
    global selected_algo
    stop_animation()
    selected_algo = label
    status_text.set_text(f"Selected: {selected_algo}")
    fig.canvas.draw_idle()

radio.on_clicked(on_select)

# ============================================================
# Animation
# ============================================================

def start_animation(_):
    stop_animation()
    global anim, is_running

    elapsed = benchmark_algorithm(selected_algo, data)

    working = data.copy()
    frames = ALGO_BUILDERS[selected_algo](working)

    is_running = True
    status_text.set_text(f"Running: {selected_algo}")

    def update(frame):
        draw_array(frame)
        return bars

    anim = FuncAnimation(fig, update, frames=frames,
                         interval=15, repeat=False)

    if anim.event_source:
        old_stop = getattr(anim.event_source, "stop", None)
        def wrapped_stop():
            global is_running
            is_running = False
            status_text.set_text(
                f"Done: {selected_algo} | Time: {elapsed:.6f} sec"
            )
            fig.canvas.draw_idle()
            if callable(old_stop):
                old_stop()
        anim.event_source.stop = wrapped_stop

# ============================================================
# Controls
# ============================================================

def reset(_):
    stop_animation()
    regenerate_data(len(data))
    status_text.set_text("Reset")


def new_data(_):
    stop_animation()
    regenerate_data(len(data))
    status_text.set_text("New Data")


def show_graph(results, title):
    fig2, ax2 = plt.subplots()
    algos = list(results.keys())
    times = [results[a] for a in algos]

    ax2.bar(algos, times)
    ax2.set_title(title)
    ax2.set_ylabel("Average Time (seconds)")

    for i, v in enumerate(times):
        ax2.text(i, v, f"{v:.6f}", ha='center', va='bottom')

    plt.show()


def compare_algorithms(_):
    if is_running:
        return
    results = benchmark_scientific(data, SCIENTIFIC_TRIALS)
    show_graph(results,
               f"Algorithm Comparison ({SCIENTIFIC_TRIALS} Trial Average)")

# ============================================================
# Event Bindings
# ============================================================

btn_start.on_clicked(start_animation)
btn_reset.on_clicked(reset)
btn_new.on_clicked(new_data)
btn_compare.on_clicked(compare_algorithms)

status_text.set_text("Selected: Bubble")

plt.show()
