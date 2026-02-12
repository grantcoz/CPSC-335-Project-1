import random
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def BubbleSort(arr):
    a = arr.copy()
    n = len(a)
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if a[j] > a[j + 1]:
                a[j], a[j + 1] = a[j + 1], a[j]
                swapped = True
                yield a
        if not swapped:
            break

def MergeSort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        L = arr[:mid]
        R = arr[mid:]

        MergeSort(L)
        MergeSort(R)

        i = j = k = 0
        while i < len(L) and j < len(R):
            if L[i] < R[j]:
                arr[k] = L[i]; i += 1
            else:
                arr[k] = R[j]; j += 1
            k += 1

        while i < len(L):
            arr[k] = L[i]; i += 1; k += 1
        while j < len(R):
            arr[k] = R[j]; j += 1; k += 1
    return arr

def QuickSort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return QuickSort(left) + middle + QuickSort(right)

def counting_sort(arr, exp):
    n = len(arr)
    output = [0] * n
    count = [0] * 10

    for i in range(n):
        digit = (arr[i] // exp) % 10
        count[digit] += 1

    for i in range(1, 10):
        count[i] += count[i - 1]

    i = n - 1
    while i >= 0:
        digit = (arr[i] // exp) % 10
        output[count[digit] - 1] = arr[i]
        count[digit] -= 1
        i -= 1

    for i in range(n):
        arr[i] = output[i]

def RadixSort(arr):
    if not arr:
        return arr
    if min(arr) < 0:
        raise ValueError("RadixSort here assumes non-negative integers only.")
    max_num = max(arr)
    exp = 1
    while max_num // exp > 0:
        counting_sort(arr, exp)
        exp *= 10
    return arr

def LinearSearch(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1

# --- timing helper ---
def time_call(fn, data, repeats=3):
    best = float("inf")
    for _ in range(repeats):
        arr = data.copy()
        t0 = time.perf_counter()
        out = fn(arr)  # some return new list, some sort in place; either is fine for timing
        _ = out
        best = min(best, time.perf_counter() - t0)
    return best

if __name__ == "__main__":
    
    data = [random.randint(1, 100) for _ in range(30)]
   
    # data = [64, 34, 25, 12, 22, 11, 90]
    steps = BubbleSort(data)

    fig, ax = plt.subplots()
    bars = ax.bar(range(len(data)), data)
    ax.set_title("Bubble Sort Visualization")
    ax.set_ylim(0, max(data) * 1.1)
    
    final_state = data.copy()
    
    def update(_):
        global final_state
        nxt = next(steps, None)     # None means “no more steps”
        if nxt is not None:
            final_state = nxt
        # always draw final_state (sorted at the end)
        for bar, val in zip(bars, final_state):
            bar.set_height(val)
        return bars

    anim = FuncAnimation(fig, func=lambda _: update(next(steps, data)),
                         frames=200, interval=50, blit=False, repeat=False)
    plt.show()
    
    