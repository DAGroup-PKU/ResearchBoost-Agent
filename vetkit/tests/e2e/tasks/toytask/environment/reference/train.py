"""Toy workload: a deterministic loop whose result goes to out.json. The reference computes it twice (the toy's
inefficiency: the second pass only re-checks the first)."""
import json
import sys

N = 1500000


def work(n):
    acc = 0
    for i in range(n):
        acc = (acc * 31 + i) % 1000003
    return acc


acc = work(N)
assert work(N) == acc
json.dump({'acc': acc}, open(sys.argv[1], 'w'))
