import random                                # generate random numbers for test datasets
import time                                  # high-resolution timing for benchmarks
import statistics                             # mean/stdev calculations for analysis
import matplotlib.pyplot as plt               # plotting/figure creation
from matplotlib.animation import FuncAnimation # runs frame-by-frame animations in matplotlib
from matplotlib.widgets import Button, RadioButtons, Slider  # UI widgets (buttons, radio, slider)

# ============================================================
# Algorithms (Generators for animation)
# ============================================================

def Bubblesort(arr):                          # define Bubble Sort as a generator (yields intermediate states)
    """
    Bubble Sort (generator for visualization)

    Logic:
    - Repeatedly scan the list and swap adjacent out-of-order elements.
    - After each pass, the largest remaining element bubbles to the end.
    - Optimization: if a full pass makes no swaps, the list is sorted.

    Big-O:
    - Best:    O(n)    (already sorted with early-exit)
    - Average: O(n^2)
    - Worst:   O(n^2)

    Space:
    - O(1) auxiliary (in-place). For animation we yield copies (extra memory for frames).
    """
    a = arr.copy()                            # copy input so we don’t mutate the caller’s list
    n = len(a)                                # store number of elements for loop bounds
    for i in range(n):                        # outer pass loop (each pass pushes one large value to end)
        swapped = False                       # flag to detect if this pass made any swaps
        for j in range(0, n - i - 1):         # compare adjacent pairs up to the unsorted boundary
            if a[j] > a[j + 1]:               # if two neighbors are out of order
                a[j], a[j + 1] = a[j + 1], a[j]  # swap them
                swapped = True                # record that the list was modified this pass
                yield a.copy()                # yield snapshot for animation (bars update from this list)
        if not swapped:                       # if no swaps happened in this pass
            break                             # list is already sorted, stop early

def MergeSort(arr, start=0, end=None):        # define Merge Sort as a generator over an index range
    """
    Merge Sort (generator for visualization)

    Logic:
    - Divide array into halves until size 1, then merge back in sorted order.
    - Yield after each write-back to the main array.

    Big-O:
    - Best/Average/Worst: O(n log n)

    Space:
    - O(n) extra for temporary subarrays.
    """
    if end is None:                           # if caller didn’t provide end index
        end = len(arr)                        # set end to full array length
    if end - start <= 1:                      # base case: 0 or 1 element in this segment
        return                                # already sorted; stop recursion

    mid = (start + end) // 2                  # midpoint index for splitting the segment
    yield from MergeSort(arr, start, mid)     # recursively sort left half and forward its yields
    yield from MergeSort(arr, mid, end)       # recursively sort right half and forward its yields

    left = arr[start:mid]                     # copy left half values (temporary buffer)
    right = arr[mid:end]                      # copy right half values (temporary buffer)

    i = j = 0                                 # i indexes left[], j indexes right[]
    k = start                                 # k writes back into arr[start:end]

    while i < len(left) and j < len(right):   # merge while both halves still have elements
        if left[i] < right[j]:                # if next left element is smaller
            arr[k] = left[i]                  # write it into the main array
            i += 1                            # advance left pointer
        else:                                 # otherwise right element is smaller/equal
            arr[k] = right[j]                 # write right element into the main array
            j += 1                            # advance right pointer
        k += 1                                # advance write position in main array
        yield arr.copy()                      # yield snapshot so animation shows merge progress

    while i < len(left):                      # if any elements remain in left half
        arr[k] = left[i]                      # copy remaining left elements into main array
        i += 1                                # advance left pointer
        k += 1                                # advance write pointer
        yield arr.copy()                      # yield snapshot after each write

    while j < len(right):                     # if any elements remain in right half
        arr[k] = right[j]                     # copy remaining right elements into main array
        j += 1                                # advance right pointer
        k += 1                                # advance write pointer
        yield arr.copy()                      # yield snapshot after each write

def QuickSort(arr, lo=0, hi=None):            # define in-place Quick Sort as a generator
    """
    Quick Sort (in-place generator for visualization)

    Logic:
    - Partition around a pivot (last element).
    - Recursively quicksort left and right partitions.
    - Yield after swaps so the animation shows changes.

    Big-O:
    - Best:    O(n log n)
    - Average: O(n log n)
    - Worst:   O(n^2) (bad pivots repeatedly)

    Space:
    - Recursion: O(log n) average, O(n) worst.
    """
    a = arr                                   # alias arr to a (no copy; quicksort modifies this list)
    if hi is None:                            # if no high index provided
        hi = len(a) - 1                       # set hi to last valid index

    def partition(l, r):                      # inner generator function: partitions a[l:r] around pivot
        pivot = a[r]                          # choose last element as pivot value
        i = l - 1                             # i tracks end of “<= pivot” region
        for j in range(l, r):                 # scan elements from l to r-1
            if a[j] <= pivot:                 # if current element belongs on left side
                i += 1                        # expand left region
                if i != j:                    # if element is not already in correct region position
                    a[i], a[j] = a[j], a[i]   # swap element into left region
                    yield a.copy()            # yield snapshot to show swap in animation
        if i + 1 != r:                        # if pivot is not already in correct place
            a[i + 1], a[r] = a[r], a[i + 1]   # swap pivot into final position
            yield a.copy()                    # yield snapshot after pivot swap
        return i + 1                          # return pivot’s final index (split point)

    if lo < hi:                               # only sort if segment has at least 2 elements
        gen = partition(lo, hi)               # create the partition generator for this segment
        pivot_index = None                    # will hold the returned pivot index
        try:
            while True:                       # repeatedly pull yielded frames from partition()
                yield next(gen)               # yield each swap state outward to the animation
        except StopIteration as e:            # when partition() finishes, it raises StopIteration
            pivot_index = e.value             # StopIteration.value contains partition()'s return value

        if pivot_index is None:               # safety fallback (shouldn’t normally happen)
            pivot_index = lo                  # default split

        yield from QuickSort(a, lo, pivot_index - 1)  # recursively sort left partition
        yield from QuickSort(a, pivot_index + 1, hi)  # recursively sort right partition

def RadixSort(arr):                           # define Radix Sort (LSD base 10) as a generator
    """
    Radix Sort (LSD base-10) for non-negative integers (generator for visualization)

    Logic:
    - Sort by digits from least significant to most significant.
    - Stable counting sort per digit place.
    - Yield after each write-back to show progress.

    Big-O:
    - O(d*(n + b)) ~ O(d*n), where d = number of digits, b = base (10)

    Space:
    - O(n + b)
    """
    a = arr.copy()                            # work on a copy so original array stays unchanged
    if not a:                                 # if list is empty
        return                                # nothing to sort; stop generator
    if min(a) < 0:                            # radix LSD here assumes non-negative numbers
        raise ValueError("Radix sort here assumes non-negative integers only.")  # fail fast

    def counting(exp):                        # stable counting sort by digit at exponent exp
        n = len(a)                            # number of elements
        output = [0] * n                      # output array for stable reorder
        count = [0] * 10                      # digit frequency array for digits 0..9

        for v in a:                           # count digit occurrences for this digit place
            digit = (v // exp) % 10           # extract current digit (ones, tens, hundreds, ...)
            count[digit] += 1                 # increment digit frequency

        for i in range(1, 10):                # convert frequencies to prefix sums (positions)
            count[i] += count[i - 1]          # count[i] becomes ending index for digit i

        for i in range(n - 1, -1, -1):        # traverse backwards to keep stability
            digit = (a[i] // exp) % 10        # compute digit of element being placed
            output[count[digit] - 1] = a[i]   # place element at its stable position
            count[digit] -= 1                 # decrement next position for that digit

        for i in range(n):                    # write output back into a
            a[i] = output[i]                  # commit sorted-by-this-digit ordering
            yield a.copy()                    # yield snapshot after each write-back

    max_num = max(a)                          # max value determines number of digit passes
    exp = 1                                   # start at ones place (10^0)
    while max_num // exp > 0:                 # while there are still digits to process
        yield from counting(exp)              # counting sort by current digit place
        exp *= 10                             # move to next digit place (10^1, 10^2, ...)

def LinearSearch(arr, target):                # define Linear Search for analysis comparison
    """
    Linear Search (used for analysis comparison, not sorting)

    Big-O:
    - Best: O(1)
    - Avg/Worst: O(n)

    Space: O(1)
    """
    for i, v in enumerate(arr):               # iterate index and value through list
        if v == target:                       # check if current value matches target
            return i                          # return found index immediately
    return -1                                 # return -1 if target never found

# Map algorithm names (shown in UI) to generator builders
ALGO_BUILDERS = {                             # dictionary: algorithm name -> function that returns frames generator
    "Bubble": lambda data: Bubblesort(data),  # Bubble frames generator for given list
    "Merge":  lambda data: MergeSort(data),   # Merge frames generator for given list
    "Quick":  lambda data: QuickSort(data, 0, len(data) - 1),  # Quick frames generator with full bounds
    "Radix":  lambda data: RadixSort(data),   # Radix frames generator for given list
}

COMPLEXITY = {                                # Big-O + space summary strings for analysis display/printing
    "Bubble": "Best O(n) (early-exit), Avg/Worst O(n^2), Space O(1)",             # bubble complexity summary
    "Merge":  "O(n log n) all cases, Space O(n)",                                 # merge complexity summary
    "Quick":  "Best/Avg O(n log n), Worst O(n^2), Space ~O(log n) avg",           # quick complexity summary
    "Radix":  "O(d·(n+b)) ~ O(d·n), Space O(n+b) (non-negative ints)",            # radix complexity summary
    "Linear (present)": "Best O(1), Avg/Worst O(n), Space O(1)",                  # linear search present-case summary
    "Linear (absent)":  "Best O(1), Avg/Worst O(n), Space O(1)",                  # linear search absent-case summary
}

# ============================================================
# Benchmarking / Comparative Analysis
# ============================================================

SCIENTIFIC_TRIALS = 5                         # how many runs to average for benchmarking
LOW, HIGH = 1, 100                            # value range for generated dataset elements

def benchmark_sort_algorithm(algo_name, dataset):  # benchmark one sorting algorithm on a given dataset
    """Time a sorting algorithm by consuming its generator frames."""
    test_data = dataset.copy()                # copy dataset so benchmarking doesn’t mutate the original
    gen = ALGO_BUILDERS[algo_name](test_data) # create generator for the chosen algorithm
    start = time.perf_counter()               # start high-resolution timer
    for _ in gen:                             # consume all frames until sorting finishes
        pass                                  # do nothing per-frame; timing is what matters
    return time.perf_counter() - start        # return elapsed seconds

def benchmark_linear_search_samples(dataset, trials=SCIENTIFIC_TRIALS):  # collect timing samples for search
    """Collect samples for linear search (present and absent)."""
    if dataset:                               # if dataset is non-empty
        present_target = dataset[len(dataset) // 2]  # choose a guaranteed present element (middle)
        absent_target = max(dataset) + 1      # choose a value guaranteed absent (outside max)
    else:                                     # if dataset is empty
        present_target = 0                    # arbitrary target (no element exists)
        absent_target = 1                     # arbitrary target (still absent)

    def run_one(target):                      # helper to time one linear search run
        t0 = time.perf_counter()              # start timer
        _ = LinearSearch(dataset, target)     # perform the search
        return time.perf_counter() - t0       # return elapsed seconds

    return {                                  # return dict of sample lists for both cases
        "Linear (present)": [run_one(present_target) for _ in range(trials)],  # times when found
        "Linear (absent)":  [run_one(absent_target) for _ in range(trials)],   # times when not found
    }

def benchmark_scientific_stats(dataset, trials=SCIENTIFIC_TRIALS, include_linear=True):  # compute stats
    """Return per-algorithm timing stats (mean/stdev/min/max) over multiple trials."""
    samples = {}                              # map algorithm -> list of timing runs
    for name in ALGO_BUILDERS:                # for each sorting algorithm
        samples[name] = [benchmark_sort_algorithm(name, dataset) for _ in range(trials)]  # collect times

    if include_linear:                        # optionally include linear search samples too
        samples.update(benchmark_linear_search_samples(dataset, trials=trials))  # merge search runs

    stats = {}                                # map algorithm -> computed statistics
    for name, runs in samples.items():        # iterate over run lists
        mean = statistics.fmean(runs) if runs else 0.0  # compute mean time (0.0 if empty)
        stdev = statistics.pstdev(runs) if len(runs) > 1 else 0.0  # population stdev (0 if single)
        stats[name] = {                       # store summary statistics and raw runs
            "mean": mean,                     # average time
            "stdev": stdev,                   # variability across trials
            "min": min(runs) if runs else 0.0,# best (fastest) observed run
            "max": max(runs) if runs else 0.0,# worst (slowest) observed run
            "runs": runs,                     # the raw timing samples (for debugging/reporting)
        }
    return stats                               # return all stats

def print_analysis_summary(stats, condition, n, trials=SCIENTIFIC_TRIALS):  # print benchmark summary
    ranked = sorted(stats.items(), key=lambda kv: kv[1]["mean"])  # sort algorithms by mean time ascending
    fastest_name, fastest = ranked[0]          # first item is fastest
    slowest_name, slowest = ranked[-1]         # last item is slowest

    print("\n=== Performance Summary ===")      # header for console output
    print(f"n={n} | condition={condition} | trials={trials}")  # show dataset size, condition, trials
    print(f"Fastest: {fastest_name} ({fastest['mean']:.6f}s)") # print fastest algorithm and mean time
    print(f"Slowest: {slowest_name} ({slowest['mean']:.6f}s)") # print slowest algorithm and mean time
    print("\nRanked results (mean ± stdev) and speedup vs slowest:")  # explain ranking columns
    for name, s in ranked:                     # print each algorithm in ranked order
        speedup = (slowest["mean"] / s["mean"]) if s["mean"] > 0 else float("inf")  # speedup ratio
        print(f"- {name:16} {s['mean']:.6f} ± {s['stdev']:.6f}   ({speedup:6.1f}x)")  # formatted line

    print("\nTheory (Big-O):")                 # header for complexity section
    for name, _ in ranked:                     # list complexities in ranked order
        if name in COMPLEXITY:                 # if complexity summary exists for this algorithm
            print(f"- {name:16} {COMPLEXITY[name]}")  # print complexity text

def show_graph_detailed(stats, title):         # draw bar graph of benchmark means with error bars
    """Bar chart with error bars + Big-O legend box (placed to the right, no overlap)."""
    # Wider figure so x-labels and the complexity box fit cleanly
    fig2, ax2 = plt.subplots(figsize=(11, 5))  # create a new figure for the comparison plot

    algos = list(stats.keys())                 # list of algorithm labels (dict insertion order)
    means = [stats[a]["mean"] for a in algos]  # mean times for each algorithm
    errs  = [stats[a]["stdev"] for a in algos] # stdev values for error bars

    ax2.bar(range(len(algos)), means, yerr=errs, capsize=4)  # bar plot with error bars
    ax2.set_title(title, pad=10)              # set plot title with padding
    ax2.set_ylabel("Average Time (seconds)")  # label y-axis

    ax2.set_xticks(range(len(algos)))         # set tick positions for each bar
    ax2.set_xticklabels(algos, rotation=35, ha="right")  # rotate labels for readability

    # Add value labels
    offset = max(means) * 0.02 if max(means) > 0 else 0.000001  # vertical offset above bars
    for i, v in enumerate(means):             # loop through each bar index/value
        ax2.text(i, v + offset, f"{v:.6f}", ha="center", va="bottom")  # annotate bar height

    # Reserve right margin for the Big-O box
    fig2.subplots_adjust(right=0.70)          # shrink plot width so text box can sit outside

    legend_lines = []                         # list of text lines describing each algorithm’s complexity
    for name in algos:                        # for each algorithm label
        if name in COMPLEXITY:                # if complexity data exists
            legend_lines.append(f"{name}: {COMPLEXITY[name]}")  # add formatted complexity line

    if legend_lines:                          # if we have lines to display
        text_block = "\n".join(legend_lines)  # join them into one multi-line string
        ax2.text(                              # place text relative to axis coordinates
            1.02, 0.98,                        # x,y beyond right edge (1.02) and near top (0.98)
            text_block,                        # complexity text content
            transform=ax2.transAxes,           # interpret coordinates in axis fraction units
            va="top",                          # align text box to top
            fontsize=9,                        # set readable font size
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.90)  # background box styling
        )

    ax2.grid(True, axis="y", alpha=0.25)      # add light horizontal grid lines for readability
    plt.show()                                # display the comparison figure window

def show_scaling_plot(condition, n_values, trials=SCIENTIFIC_TRIALS):  # plot runtime vs n
    """
    Plot time vs n for each sorting algorithm for one condition.
    Bubble is automatically skipped for very large n to avoid long runs.
    """
    fig3, ax3 = plt.subplots()                # create a new figure for scaling curves
    for algo in ["Bubble", "Merge", "Quick", "Radix"]:  # choose which algorithms to include
        xs, ys = [], []                       # xs = n values used; ys = mean times for those n
        for n in n_values:                    # loop through tested sizes
            if algo == "Bubble" and n > 1200: # avoid extremely slow bubble sort for large n
                continue                      # skip these n values for bubble
            dataset = generate_dataset(n, condition)  # create dataset of this size and condition
            runs = [benchmark_sort_algorithm(algo, dataset) for _ in range(trials)]  # collect times
            xs.append(n)                      # record size
            ys.append(statistics.fmean(runs)) # record average runtime
        ax3.plot(xs, ys, marker="o", label=algo)  # plot one line per algorithm

    ax3.set_title(f"Scaling (mean of {trials} trials) | {condition}")  # title shows trials and condition
    ax3.set_xlabel("n (array size)")          # x-axis label
    ax3.set_ylabel("Time (seconds)")          # y-axis label
    ax3.legend()                              # show legend mapping lines to algorithms
    ax3.grid(True, alpha=0.3)                 # show light grid for readability
    plt.tight_layout()                        # adjust spacing so labels fit
    plt.show()                                # display scaling figure window

# ============================================================
# Data generation (random / sorted / reverse)
# ============================================================

def generate_dataset(n, condition):           # create a dataset based on the chosen condition
    if condition == "Random":                 # if random condition selected
        return [random.randint(LOW, HIGH) for _ in range(n)]  # n random integers in range [LOW,HIGH]
    if condition == "Sorted":                 # if sorted condition selected
        return sorted([random.randint(LOW, HIGH) for _ in range(n)])  # generate then sort ascending
    if condition == "Reverse":                # if reverse condition selected
        return sorted([random.randint(LOW, HIGH) for _ in range(n)], reverse=True)  # sort descending
    return [random.randint(LOW, HIGH) for _ in range(n)]  # fallback: random list

# ============================================================
# UI + Animation
# ============================================================

N0 = 35                                       # default starting array size for visualization
data_condition = "Random"                     # default data condition for visualization

data = generate_dataset(N0, data_condition)   # create initial dataset
original_data = data.copy()                   # store a copy so Reset can restore this exact dataset

selected_algo = "Bubble"                      # default selected algorithm in the UI
anim = None                                   # will hold the FuncAnimation instance when running
is_running = False                            # True while animation is active
is_paused = False                             # True when paused

fig, ax = plt.subplots()                      # create main window figure + axes for bars
# Leave a wider left margin for controls + slider label space
plt.subplots_adjust(left=0.38, right=0.98, top=0.92, bottom=0.28)  # allocate room for widgets

bars = ax.bar(range(len(data)), data)         # draw initial bar chart (one bar per array element)
ax.set_ylim(0, HIGH * 1.1)                    # set y-axis limit based on max possible value
ax.set_title("Sorting Visualizer")            # set title of the main visualization
status_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")  # status overlay text

# Slider for number of elements (moved right so it never overlaps controls)
ax_slider = plt.axes([0.38, 0.14, 0.58, 0.04])# create an axis region for the slider widget
slider = Slider(ax_slider, "n", 5, 200, valinit=len(data), valstep=5)  # slider controls n (step 5)

# Algorithm radio buttons
rax_algo = plt.axes([0.05, 0.58, 0.26, 0.32]) # create axis region for algorithm selector
radio_algo = RadioButtons(rax_algo, list(ALGO_BUILDERS.keys()))  # options from ALGO_BUILDERS keys
radio_algo.set_active(0)                      # select first option ("Bubble") by default

# Condition radio buttons
rax_cond = plt.axes([0.05, 0.44, 0.26, 0.12]) # create axis region for condition selector
radio_cond = RadioButtons(rax_cond, ["Random", "Sorted", "Reverse"])  # dataset conditions
radio_cond.set_active(0)                      # select first option ("Random") by default

# Buttons
ax_start = plt.axes([0.05, 0.36, 0.26, 0.06]) # axis region for Start button
btn_start = Button(ax_start, "Start")         # create Start button widget

ax_pause = plt.axes([0.05, 0.29, 0.26, 0.06]) # axis region for Pause/Resume button
btn_pause = Button(ax_pause, "Pause")         # create Pause button widget (label changes to Resume)

ax_reset = plt.axes([0.05, 0.22, 0.26, 0.06]) # axis region for Reset button
btn_reset = Button(ax_reset, "Reset")         # create Reset button widget

ax_new = plt.axes([0.05, 0.15, 0.26, 0.06])   # axis region for New Data button
btn_new = Button(ax_new, "New Data")          # create New Data button widget

ax_compare = plt.axes([0.05, 0.08, 0.26, 0.06]) # axis region for Compare button
btn_compare = Button(ax_compare, "Compare")     # create Compare button widget (benchmarks + bar chart)

ax_scale = plt.axes([0.05, 0.00, 0.26, 0.06])   # axis region for Compare Scaling button
btn_scale = Button(ax_scale, "Compare Scaling") # create Compare Scaling button widget

# -----------------------------
# Helpers
# -----------------------------

def stop_animation():                         # stop any active animation and reset state flags
    """Stop any running animation cleanly (used when changing settings)."""
    global anim, is_running, is_paused        # declare that we modify global variables
    if anim is not None and getattr(anim, "event_source", None) is not None:  # if animation exists
        anim.event_source.stop()              # stop the timer that drives animation frames
    anim = None                               # clear animation reference
    is_running = False                        # mark not running
    is_paused = False                         # mark not paused
    btn_pause.label.set_text("Pause")         # reset button label back to Pause

def rebuild_bars(arr):                        # rebuild the bar chart when the array length changes
    """Clear axes and rebuild bars for a new length."""
    global bars, status_text                  # we replace these globals
    ax.clear()                                # clear the main axes (removes old bars)
    ax.set_ylim(0, HIGH * 1.1)                # reapply y limit after clearing
    ax.set_title("Sorting Visualizer")        # reapply title after clearing
    bars = ax.bar(range(len(arr)), arr)       # recreate bars for the new array size
    status_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")  # recreate status text
    fig.canvas.draw_idle()                    # request a redraw

def redraw_bars(arr):                         # update bar heights without rebuilding bars
    for bar, val in zip(bars, arr):           # pair each bar with its new value
        bar.set_height(val)                   # set bar height to match array value

def apply_condition_and_size(n):              # regenerate dataset and rebuild chart for new n/condition
    """Recreate data using current condition and size, update original_data, rebuild bars."""
    global data, original_data                # update shared dataset variables
    data = generate_dataset(n, data_condition)# generate new dataset based on current condition
    original_data = data.copy()               # save snapshot for Reset
    rebuild_bars(data)                        # rebuild bars for potentially new size

# -----------------------------
# UI callbacks
# -----------------------------

def on_slider_change(_val):                   # called whenever slider moves
    if is_running:                            # if animation is currently running
        stop_animation()                      # stop it before changing dataset size
    new_n = int(slider.val)                   # get slider value and convert to integer
    apply_condition_and_size(new_n)           # regenerate dataset and rebuild bars
    status_text.set_text(f"Size: {new_n} | Condition: {data_condition} | Algo: {selected_algo}")  # update status
    fig.canvas.draw_idle()                    # refresh display

slider.on_changed(on_slider_change)           # connect slider changes to callback

def on_select_algo(label):                    # called when user picks a different algorithm
    global selected_algo                      # we update selected_algo
    if is_running:                            # if currently running
        stop_animation()                      # stop before switching algorithms
    selected_algo = label                     # store chosen algorithm name
    status_text.set_text(f"Selected Algo: {selected_algo}")  # show selection in status text
    fig.canvas.draw_idle()                    # redraw UI

radio_algo.on_clicked(on_select_algo)         # connect algorithm radio selection to callback

def on_select_condition(label):               # called when user picks Random/Sorted/Reverse
    global data_condition                     # update global condition
    if is_running:                            # if animation is running
        stop_animation()                      # stop it before changing dataset condition
    data_condition = label                    # store new condition
    apply_condition_and_size(int(slider.val)) # regenerate data with same n but new condition
    status_text.set_text(f"Condition: {data_condition} | n={len(data)} | Algo: {selected_algo}")  # update status
    fig.canvas.draw_idle()                    # redraw UI

radio_cond.on_clicked(on_select_condition)    # connect condition radio selection to callback

def start_animation(_event):                  # called when Start button is clicked
    global anim, is_running, is_paused        # modify global animation state variables

    if is_running:                            # if already running
        return                                # ignore extra start clicks

    stop_animation()                          # ensure a clean start state
    is_running = True                         # mark animation as running
    is_paused = False                         # mark not paused
    btn_pause.label.set_text("Pause")         # ensure pause button label is correct

    working = data.copy()                     # copy current data for this run (don’t mutate stored data)
    gen = ALGO_BUILDERS[selected_algo](working) # create algorithm frame generator
    start_t = time.perf_counter()             # record start time for “Done” message

    def update(arr_state):                    # per-frame update function called by FuncAnimation
        redraw_bars(arr_state)                # update bar heights to match this frame’s array state
        return bars                           # return artists (bars) to matplotlib

    def on_finish():                          # called when generator is exhausted (sorting completes)
        elapsed = time.perf_counter() - start_t  # compute total elapsed time for this animation run
        global is_running, is_paused          # update global flags
        is_running = False                    # mark not running
        is_paused = False                     # mark not paused
        btn_pause.label.set_text("Pause")     # reset pause button label
        status_text.set_text(f"Done: {selected_algo} | {data_condition} | n={len(data)} | {elapsed:.6f}s")  # show result
        fig.canvas.draw_idle()                # redraw to show updated status

    def frames():                             # wrapper generator that triggers on_finish at the end
        for frame in gen:                     # yield each frame from the algorithm generator
            yield frame                       # forward frame to FuncAnimation
        on_finish()                           # once generator ends, update status and flags

    status_text.set_text(f"Running: {selected_algo} | {data_condition} | n={len(data)}")  # show running status
    anim = FuncAnimation(fig, update, frames=frames(), interval=50, repeat=False)  # create the animation
    fig.canvas.draw_idle()                    # request a redraw

def toggle_pause(_event):                     # called when Pause/Resume button is clicked
    global is_paused, is_running, anim        # modify pause state and animation reference
    if anim is None or anim.event_source is None:  # if no animation exists
        return                                # nothing to pause/resume
    if not is_running:                        # if not currently running
        return                                # do nothing

    if not is_paused:                         # if currently running (not paused)
        anim.event_source.stop()              # stop the timer that drives frames (pauses)
        is_paused = True                      # record paused state
        btn_pause.label.set_text("Resume")    # change button label to Resume
        status_text.set_text(f"Paused: {selected_algo} | {data_condition} | n={len(data)}")  # update status
    else:                                     # if currently paused
        anim.event_source.start()             # restart timer (resume animation)
        is_paused = False                     # record running state
        btn_pause.label.set_text("Pause")     # change button label back to Pause
        status_text.set_text(f"Running: {selected_algo} | {data_condition} | n={len(data)}")  # update status

    fig.canvas.draw_idle()                    # redraw to reflect label/status changes

def reset(_event):                            # called when Reset button is clicked
    global data                               # reset the global dataset
    stop_animation()                          # stop any running animation
    data = original_data.copy()               # restore the original dataset snapshot
    rebuild_bars(data)                        # rebuild bars (safe even if same length)
    status_text.set_text(f"Reset | Algo: {selected_algo} | {data_condition} | n={len(data)}")  # update status
    fig.canvas.draw_idle()                    # redraw UI

def new_data(_event):                         # called when New Data button is clicked
    stop_animation()                          # stop animation before replacing dataset
    apply_condition_and_size(int(slider.val)) # regenerate dataset using current n and condition
    status_text.set_text(f"New data | Algo: {selected_algo} | {data_condition} | n={len(data)}")  # update status
    fig.canvas.draw_idle()                    # redraw UI

def compare_algorithms(_event):               # called when Compare button is clicked
    """Compare algorithms on current dataset; include linear search timings too."""
    if is_running:                            # if animation is running
        return                                # skip comparison to avoid changing state mid-run
    stats = benchmark_scientific_stats(data, trials=SCIENTIFIC_TRIALS, include_linear=True)  # gather stats
    print_analysis_summary(stats, data_condition, len(data), trials=SCIENTIFIC_TRIALS)        # print console summary
    show_graph_detailed(stats, f"Comparison ({SCIENTIFIC_TRIALS} trials) | n={len(data)} | {data_condition}")  # show plot

def compare_scaling(_event):                  # called when Compare Scaling button is clicked
    """Runs time vs n for the currently selected condition and plots scaling curves."""
    if is_running:                            # if animation is running
        stop_animation()                      # stop it before doing long benchmarks
    n_values = [50, 100, 200, 400, 800, 1200, 1600]  # array sizes to test (bubble capped inside function)
    show_scaling_plot(data_condition, n_values, trials=3)  # run scaling benchmarks and plot

# Bindings
btn_start.on_clicked(start_animation)         # connect Start button to start_animation callback
btn_pause.on_clicked(toggle_pause)            # connect Pause button to toggle_pause callback
btn_reset.on_clicked(reset)                   # connect Reset button to reset callback
btn_new.on_clicked(new_data)                  # connect New Data button to new_data callback
btn_compare.on_clicked(compare_algorithms)    # connect Compare button to compare_algorithms callback
btn_scale.on_clicked(compare_scaling)         # connect Compare Scaling to compare_scaling callback

status_text.set_text(f"Ready | Algo: {selected_algo} | Condition: {data_condition} | n={len(data)}")  # initial status
plt.show()                                    # start matplotlib event loop; opens the window
