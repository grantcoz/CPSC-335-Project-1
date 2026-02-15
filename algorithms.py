import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import RadioButtons, Button

# -----------------------------
# Sorting frame generators
# -----------------------------

"""
Bubble Sort

Logic:
- Repeatedly scan the list.
- Swap adjacent elements that are out of order.
- After each pass, the largest remaining element "bubbles" to the end.
- Optimization: if a full pass makes no swaps, it is already sorted.

Big-O:
- Best:    O(n)    (already sorted; no swaps, early exit)
- Average: O(n^2)
- Worst:   O(n^2)  (reverse sorted)

Space:
- Typical bubble sort is O(1) extra space, but here we yield copies for
    animation, so we allocate extra memory for frames.
"""
def Bubblesort(arr):
    # Work on a copy so caller's array is not modified
    a = arr.copy()
    # Store length for loops
    n = len(a)

    # Outer loop performs passes through the array
    for i in range(n):
        # Flag to detect whether any swap happened this pass
        swapped = False

        # Inner loop compares adjacent pairs up to the unsorted boundary
        for j in range(0, n - i - 1):
            # If a pair is out of order, swap it
            if a[j] > a[j + 1]:
                a[j], a[j + 1] = a[j + 1], a[j]
                swapped = True
                # Yield a snapshot for the animation frame
                yield a.copy()

        # If no swaps happened, array is sorted; stop early (best-case O(n))
        if not swapped:
            break

"""
Merge Sort

Logic (Divide & Conquer):
1) Split array into two halves until subarrays of size 1.
2) Merge halves back together in sorted order.
3) During merge, we write values back into 'arr' and yield after each write.

Big-O:
- Best:    O(n log n)
- Average: O(n log n)
- Worst:   O(n log n)

Space:
- O(n) extra for temporary left/right arrays during merging.
"""
def MergeSort(arr, start=0, end=None):
    # If end not provided, set it to length of array
    if end is None:
        end = len(arr)

    # Base case: a slice of length 0 or 1 is already sorted
    if end - start <= 1:
        return

    # Compute midpoint for splitting
    mid = (start + end) // 2

    # Recursively sort left half, yielding its frames
    yield from MergeSort(arr, start, mid)
    # Recursively sort right half, yielding its frames
    yield from MergeSort(arr, mid, end)

    # Copy out the left half
    left = arr[start:mid]
    # Copy out the right half
    right = arr[mid:end]

    # i = index into left, j = index into right
    i = j = 0
    # k = index into original array segment
    k = start

    # Merge until one side is exhausted
    while i < len(left) and j < len(right):
        # Take the smaller front element
        if left[i] < right[j]:
            arr[k] = left[i]
            i += 1
        else:
            arr[k] = right[j]
            j += 1
        k += 1
        # Yield snapshot after each write (shows merge progress)
        yield arr.copy()

    # Copy any remaining left elements
    while i < len(left):
        arr[k] = left[i]
        i += 1
        k += 1
        yield arr.copy()

    # Copy any remaining right elements
    while j < len(right):
        arr[k] = right[j]
        j += 1
        k += 1
        yield arr.copy()

"""
Quick Sort (Generator, in-place)

Logic:
- Choose pivot (here: last element).
- Partition array so:
    <= pivot on the left, > pivot on the right.
- Recursively quicksort left and right partitions.
- Yield after each swap so animation shows changes.

Big-O:
- Best:    O(n log n)  (balanced partitions)
- Average: O(n log n)
- Worst:   O(n^2)      (bad pivot repeatedly; e.g., sorted input with last pivot)

Space:
- Recursion stack: O(log n) average, O(n) worst.
"""
def QuickSort(arr, lo=0, hi=None):
    # Use the same list reference; quicksort is in-place
    a = arr

    # If hi is not set, use last index
    if hi is None:
        hi = len(a) - 1

    def partition(l, r):
        # Pick pivot as last element
        pivot = a[r]
        # i marks the boundary of elements <= pivot
        i = l - 1

        # Walk through subarray
        for j in range(l, r):
            # If current element belongs in <= pivot region
            if a[j] <= pivot:
                i += 1
                # Swap into the <= pivot region if needed
                if i != j:
                    a[i], a[j] = a[j], a[i]
                    yield a.copy()

        # Put pivot into final correct spot
        if i + 1 != r:
            a[i + 1], a[r] = a[r], a[i + 1]
            yield a.copy()

        # Return pivot index
        return i + 1

    # Only sort if there are at least two elements
    if lo < hi:
        # Create the partition generator
        gen = partition(lo, hi)
        pivot_index = None

        try:
            # Yield all frames from partition
            while True:
                frame = next(gen)
                yield frame
        except StopIteration as e:
            # Extract pivot index returned by partition
            pivot_index = e.value

        # Fallback just in case return value isn't captured
        if pivot_index is None:
            pivot_index = lo

        # Recursively sort left partition
        yield from QuickSort(a, lo, pivot_index - 1)
        # Recursively sort right partition
        yield from QuickSort(a, pivot_index + 1, hi)

"""
Radix Sort (LSD base-10) for non-negative integers (Generator)

Logic:
- Sort by digits from least significant to most significant.
- For each digit place, run a stable counting sort.
- Yield after each write-back to show progress.

Big-O:
Let:
    n = number of elements
    d = number of digits of max element
    b = base (10)
- Best:    O(d*(n + b))
- Average: O(d*(n + b))
- Worst:   O(d*(n + b))
Usually simplified to O(dn) since b=10 is constant.

Space:
- O(n + b) for output + count arrays.
"""
def RadixSort(arr):
    # Work on a copy so caller's array isn't changed unexpectedly
    a = arr.copy()

    # If empty, nothing to do
    if not a:
        return

    # This radix sort implementation assumes non-negative ints
    if min(a) < 0:
        raise ValueError("Radix sort here assumes non-negative integers only.")

    def counting(exp):
        # n = number of values
        n = len(a)
        # Temporary output array
        output = [0] * n
        # Count array for digits 0-9
        count = [0] * 10

        # Count occurrences of each digit
        for v in a:
            digit = (v // exp) % 10
            count[digit] += 1

        # Convert counts into prefix sums (positions)
        for i in range(1, 10):
            count[i] += count[i - 1]

        # Build output array from right-to-left for stability
        for i in range(n - 1, -1, -1):
            digit = (a[i] // exp) % 10
            output[count[digit] - 1] = a[i]
            count[digit] -= 1

        # Copy output back into a and yield after each write
        for i in range(n):
            a[i] = output[i]
            yield a.copy()

    # Find the maximum value to know how many digit passes we need
    max_num = max(a)
    # exp controls which digit we are sorting by (1s, 10s, 100s, ...)
    exp = 1

    # Loop over digit places until we've processed all digits
    while max_num // exp > 0:
        yield from counting(exp)
        exp *= 10

# Map algorithm names (shown in UI) to generator builders
ALGO_BUILDERS = {
    # Bubble sort builder: returns bubble generator
    "Bubble": lambda data: Bubblesort(data),
    # Merge sort builder: returns merge generator
    "Merge":  lambda data: MergeSort(data),
    # Quick sort builder: returns quicksort generator
    "Quick":  lambda data: QuickSort(data, 0, len(data) - 1),
    # Radix sort builder: returns radix generator
    "Radix":  lambda data: RadixSort(data),
}

# ============================================================
# UI + Animation
# ============================================================

# Number of bars/elements to sort
N = 35
# Minimum and maximum random values
LOW, HIGH = 1, 100

# Generate initial random dataset
data = [random.randint(LOW, HIGH) for _ in range(N)]
# Store a copy to support Reset
original_data = data.copy()

# Track which algorithm is selected
selected_algo = "Bubble"
# Will store the FuncAnimation instance
anim = None
# Flag that prevents starting or switching algorithm mid-animation
is_running = False

# Create the plot window and axis
fig, ax = plt.subplots()
# Make space on left for UI controls
plt.subplots_adjust(left=0.28, right=0.98, top=0.92, bottom=0.08)

# Create bar chart using initial data
bars = ax.bar(range(N), data)
# Title at the top
ax.set_title("Sorting Visualizer")
# Fix y-axis scale so bars don't rescale during animation
ax.set_ylim(0, HIGH * 1.1)

# Create an axes region to hold radio buttons (x, y, width, height in figure coords)
rax = plt.axes([0.04, 0.55, 0.20, 0.35])
# Create radio buttons from algorithm names
radio = RadioButtons(rax, list(ALGO_BUILDERS.keys()))
# Select the first algorithm by default
radio.set_active(0)

# Create axes for Start button
ax_start = plt.axes([0.04, 0.45, 0.20, 0.07])
# Create Start button widget
btn_start = Button(ax_start, "Start")

# Create axes for Reset button
ax_reset = plt.axes([0.04, 0.36, 0.20, 0.07])
# Create Reset button widget
btn_reset = Button(ax_reset, "Reset")

# Create axes for New Data button
ax_new = plt.axes([0.04, 0.27, 0.20, 0.07])
# Create New Data button widget
btn_new = Button(ax_new, "New Data")

# Create status text inside the plot (top-left)
status_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")


def draw_array(arr):
    # Update every bar height to match the current array state
    for bar, val in zip(bars, arr):
        bar.set_height(val)
    # Return bars so FuncAnimation can redraw them efficiently
    return bars


def on_select(label):
    # Allow modifying the global selected_algo
    global selected_algo

    # If currently running, ignore selection changes
    if is_running:
        return

    # Save the selected algorithm name
    selected_algo = label
    # Update the status label
    status_text.set_text(f"Selected: {selected_algo}")
    # Request a redraw of the figure
    fig.canvas.draw_idle()

# Register callback for radio button clicks
radio.on_clicked(on_select)

"""
Start sorting animation:
- Makes a working copy of the current data.
- Builds the chosen algorithm generator on that copy.
- FuncAnimation pulls frames from the generator until it ends.
- Wraps the animation stop() to mark is_running False and show "Done".
"""
def start_animation(_event):
    # Use global variables
    global anim, is_running, data

    # If already running, don't start again
    if is_running:
        return

    # Work on a copy so UI data baseline is stable unless reset/new pressed
    working = data.copy()

    try:
        # Create generator frames for selected algorithm
        frames = ALGO_BUILDERS[selected_algo](working)
    except Exception as e:
        # Display error if algorithm fails (ex: Radix with negatives)
        status_text.set_text(f"Error: {e}")
        fig.canvas.draw_idle()
        return

    # Mark as running
    is_running = True
    # Update status
    status_text.set_text(f"Running: {selected_algo}")

    def update(frame_arr):
        # Update bar chart for this yielded array state
        draw_array(frame_arr)
        return bars

    # Create animation: frames is a generator, repeat=False stops at end
    anim = FuncAnimation(fig, update, frames=frames, interval=50, repeat=False)

    # If animation has an event source, wrap stop() to mark completion
    if anim.event_source is not None:
        # Keep the original stop method
        old_stop = anim.event_source.stop

        def wrapped_stop():
            # Allow modifying is_running inside this nested function
            global is_running
            # Mark not running
            is_running = False
            # Update status to show completion
            status_text.set_text(f"Done: {selected_algo}")
            # Redraw
            fig.canvas.draw_idle()
            # Call the original stop
            old_stop()

        # Replace stop method with our wrapped version
        anim.event_source.stop = wrapped_stop

    # Request draw update
    fig.canvas.draw_idle()


def reset(_event):
    # Use globals
    global anim, is_running, data, original_data

    # Stop animation if it exists
    if anim is not None and anim.event_source is not None:
        anim.event_source.stop()

    # Mark not running
    is_running = False

    # Restore data from original_data
    data = original_data.copy()
    # Redraw bars
    draw_array(data)
    # Update status
    status_text.set_text(f"Reset (Selected: {selected_algo})")
    # Redraw canvas
    fig.canvas.draw_idle()


def new_data(_event):
    # Use globals
    global anim, is_running, data, original_data

    # Stop animation if it exists
    if anim is not None and anim.event_source is not None:
        anim.event_source.stop()

    # Mark not running
    is_running = False

    # Generate new random dataset
    data = [random.randint(LOW, HIGH) for _ in range(N)]
    # Save as the new "reset" baseline
    original_data = data.copy()
    # Redraw bars
    draw_array(data)
    # Update status
    status_text.set_text(f"New data (Selected: {selected_algo})")
    # Redraw canvas
    fig.canvas.draw_idle()

# Hook up Start button to start_animation callback
btn_start.on_clicked(start_animation)
# Hook up Reset button to reset callback
btn_reset.on_clicked(reset)
# Hook up New Data button to new_data callback
btn_new.on_clicked(new_data)

# Set initial status label
status_text.set_text(f"Selected: {selected_algo}")

# Display the interactive window (blocks until window closed)
plt.show()
