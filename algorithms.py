import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import RadioButtons, Button

# -----------------------------
# Sorting frame generators
# -----------------------------

def Bubblesort(arr):
    a = arr.copy()
    n = len(a)
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if a[j] > a[j + 1]:
                a[j], a[j + 1] = a[j + 1], a[j]
                swapped = True
                yield a.copy()
        if not swapped:
            break

def MergeSort(arr, start=0, end=None):
    if end is None:
        end = len(arr)
    if end - start <= 1:
        return

    mid = (start + end) // 2
    yield from MergeSort(arr, start, mid)
    yield from MergeSort(arr, mid, end)

    left = arr[start:mid]
    right = arr[mid:end]

    i = j = 0
    k = start

    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            arr[k] = left[i]
            i += 1
        else:
            arr[k] = right[j]
            j += 1
        k += 1
        yield arr.copy()

    while i < len(left):
        arr[k] = left[i]
        i += 1
        k += 1
        yield arr.copy()

    while j < len(right):
        arr[k] = right[j]
        j += 1
        k += 1
        yield arr.copy()

def QuickSort(arr, lo=0, hi=None):
    # In-place QuickSort with yields after swaps
    a = arr
    if hi is None:
        hi = len(a) - 1

    def partition(l, r):
        pivot = a[r]
        i = l - 1
        for j in range(l, r):
            if a[j] <= pivot:
                i += 1
                if i != j:
                    a[i], a[j] = a[j], a[i]
                    yield a.copy()
        if i + 1 != r:
            a[i + 1], a[r] = a[r], a[i + 1]
            yield a.copy()
        return i + 1

    if lo < hi:
        # partition yields frames; we need pivot index too
        # so we run partition manually
        gen = partition(lo, hi)
        pivot_index = None
        try:
            while True:
                frame = next(gen)
                yield frame
        except StopIteration as e:
            pivot_index = e.value  # partition return
        if pivot_index is None:
            # fallback: compute pivot index directly if python doesn't carry return (rare)
            pivot_index = lo

        yield from QuickSort(a, lo, pivot_index - 1)
        yield from QuickSort(a, pivot_index + 1, hi)

def RadixSort(arr):
    a = arr.copy()
    if not a:
        return
    if min(a) < 0:
        raise ValueError("Radix sort here assumes non-negative integers only.")

    def counting(exp):
        n = len(a)
        output = [0] * n
        count = [0] * 10

        for v in a:
            digit = (v // exp) % 10
            count[digit] += 1
        for i in range(1, 10):
            count[i] += count[i - 1]

        for i in range(n - 1, -1, -1):
            digit = (a[i] // exp) % 10
            output[count[digit] - 1] = a[i]
            count[digit] -= 1

        for i in range(n):
            a[i] = output[i]
            yield a.copy()  # yield after each write-back for animation

    max_num = max(a)
    exp = 1
    while max_num // exp > 0:
        yield from counting(exp)
        exp *= 10

# Map names -> generator builder
ALGO_BUILDERS = {
    "Bubble": lambda data: Bubblesort(data),
    "Merge":  lambda data: MergeSort(data),
    "Quick":  lambda data: QuickSort(data, 0, len(data) - 1),
    "Radix":  lambda data: RadixSort(data),
}

# -----------------------------
# UI + Animation
# -----------------------------

N = 35
LOW, HIGH = 1, 100

data = [random.randint(LOW, HIGH) for _ in range(N)]
original_data = data.copy()

selected_algo = "Bubble"
anim = None
is_running = False

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.28, right=0.98, top=0.92, bottom=0.08)

bars = ax.bar(range(N), data)
ax.set_title("Sorting Visualizer")
ax.set_ylim(0, HIGH * 1.1)

# Radio buttons panel
rax = plt.axes([0.04, 0.55, 0.20, 0.35])
radio = RadioButtons(rax, list(ALGO_BUILDERS.keys()))
radio.set_active(0)

# Buttons
ax_start = plt.axes([0.04, 0.45, 0.20, 0.07])
btn_start = Button(ax_start, "Start")

ax_reset = plt.axes([0.04, 0.36, 0.20, 0.07])
btn_reset = Button(ax_reset, "Reset")

ax_new = plt.axes([0.04, 0.27, 0.20, 0.07])
btn_new = Button(ax_new, "New Data")

status_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")

def draw_array(arr):
    for bar, val in zip(bars, arr):
        bar.set_height(val)
    return bars

def on_select(label):
    global selected_algo
    if is_running:
        return
    selected_algo = label
    status_text.set_text(f"Selected: {selected_algo}")
    fig.canvas.draw_idle()

radio.on_clicked(on_select)

def start_animation(_event):
    global anim, is_running, data

    if is_running:
        return

    # Build frames based on selection
    working = data.copy()

    try:
        frames = ALGO_BUILDERS[selected_algo](working)
    except Exception as e:
        status_text.set_text(f"Error: {e}")
        fig.canvas.draw_idle()
        return

    is_running = True
    status_text.set_text(f"Running: {selected_algo}")

    def update(frame_arr):
        # frame_arr is the yielded array state
        draw_array(frame_arr)
        return bars

    def on_done(_):
        # Not always called reliably; we’ll also stop by catching generator end
        pass

    # Important: use frames=generator so it stops when generator ends
    anim = FuncAnimation(fig, update, frames=frames, interval=50, repeat=False)

    # When the generator ends, matplotlib stops requesting frames; mark not running
    # We can detect this by hooking into draw events with a small trick:
    def mark_finished(event):
        # If animation exists but isn't actively iterating anymore,
        # this draw event typically fires at the end too.
        # We'll just allow Reset/New to work regardless.
        pass

    fig.canvas.draw_idle()

    # We can’t perfectly detect end without extra plumbing;
    # easiest: allow Reset/New at any time, and block only Start during run.
    # Also: after some time, Start will work again once animation naturally ends.
    # To make it explicit, you can just hit Reset/New to stop visually.

    # Schedule “unlock” by using the animation event source callback:
    # (when repeat=False and frames are exhausted, it stops.)
    if anim.event_source is not None:
        old_stop = anim.event_source.stop
        def wrapped_stop():
            global is_running
            is_running = False
            status_text.set_text(f"Done: {selected_algo}")
            fig.canvas.draw_idle()
            old_stop()
        anim.event_source.stop = wrapped_stop

def reset(_event):
    global anim, is_running, data, original_data

    if anim is not None and anim.event_source is not None:
        anim.event_source.stop()
    is_running = False

    data = original_data.copy()
    draw_array(data)
    status_text.set_text(f"Reset (Selected: {selected_algo})")
    fig.canvas.draw_idle()

def new_data(_event):
    global anim, is_running, data, original_data

    if anim is not None and anim.event_source is not None:
        anim.event_source.stop()
    is_running = False

    data = [random.randint(LOW, HIGH) for _ in range(N)]
    original_data = data.copy()
    draw_array(data)
    status_text.set_text(f"New data (Selected: {selected_algo})")
    fig.canvas.draw_idle()

btn_start.on_clicked(start_animation)
btn_reset.on_clicked(reset)
btn_new.on_clicked(new_data)

status_text.set_text(f"Selected: {selected_algo}")
plt.show()
