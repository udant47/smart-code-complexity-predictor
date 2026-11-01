"""Sample programs for the analyzer. They are only parsed, never run."""

SAMPLES: dict[str, str] = {
    "Merge sort (divide and conquer)": '''def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    return result + left[i:] + right[j:]


def merge_sort(items):
    if len(items) <= 1:
        return items
    mid = len(items) // 2
    return merge(merge_sort(items[:mid]), merge_sort(items[mid:]))
''',
    "Binary search": '''def binary_search(items, target):
    lo, hi = 0, len(items) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if items[mid] == target:
            return mid
        if items[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
''',
    "Pair counting (nested loops)": '''def count_zero_pairs(values):
    count = 0
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            if values[i] + values[j] == 0:
                count += 1
    return count


def has_zero_pair(values):
    seen = set()
    for v in values:
        if -v in seen:
            return True
        seen.add(v)
    return False
''',
    "Naive Fibonacci (branching recursion)": '''def fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)


def fib_iterative(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
''',
    "Graph traversal (class)": '''from collections import deque


class Graph:
    def __init__(self):
        self.adj = {}

    def add_edge(self, u, v):
        self.adj.setdefault(u, []).append(v)
        self.adj.setdefault(v, []).append(u)

    def bfs(self, start):
        seen = {start}
        order = []
        queue = deque([start])
        while queue:
            node = queue.popleft()
            order.append(node)
            for nxt in self.adj.get(node, []):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        return order
''',
    "Tangled order processing (refactoring candidate)": '''def process_orders(orders, inventory, rules, mode, verbose, strict):
    results = []
    for order in orders:
        if order is not None:
            for item in order.items:
                if item.sku in inventory:
                    if mode == "express" and strict:
                        for rule in rules:
                            if rule.applies(item):
                                while item.qty > 0 and inventory[item.sku] > 0:
                                    inventory[item.sku] -= 1
                                    item.qty -= 1
                                    if verbose:
                                        print("reserved", item.sku)
                    elif mode == "standard" or verbose:
                        results.append((order.id, item.sku))
    return results


def shipping_code(region):
    if region == "north":
        return 10
    elif region == "south":
        return 20
    elif region == "east":
        return 30
    elif region == "west":
        return 40
    elif region == "central":
        return 50
    return 0
''',
}

# A before/after pair for the comparison view (the "after" was written by hand, not generated).
COMPARE_BEFORE = SAMPLES["Pair counting (nested loops)"].split("\n\n\ndef has_zero_pair")[0] + "\n"
COMPARE_AFTER = '''def count_zero_pairs(values):
    counts = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    pairs = 0
    for v, c in counts.items():
        if v > 0:
            pairs += c * counts.get(-v, 0)
    zeros = counts.get(0, 0)
    return pairs + zeros * (zeros - 1) // 2
'''
