#! /usr/bin/python
# -*- encoding: utf-8 -*-

# dok(Dictionary of Keys format)形式の一般的な関数群

import math
import random
import bisect
from itertools import groupby
from copy import deepcopy
from collections import defaultdict

def __neg__(self):
    return dokneg(self)

def __add__(self, other):
    return doksum(self, other)

def __sub__(self,other):
    return dokdiff(self,other)

def dokneg(x):
    # 負
    new = deepcopy(x)
    for k,v in x.items():
        new[k] = -v
    return new

def doksum(*x):
    # 和
    new = deepcopy(x[0])
    for a in x[1:]:
        for k,v in a.items():
            new[k] += v
    return new

def dokdiff(a,b):
    # 差
    new = deepcopy(a)
    for k,v in b.items():
        new[k] -= v
    return new

def product_scalar(d,f):
    # スカラー積
    z = deepcopy(d)
    for k,v in d.items():
        z[k] = v * f
    return z

def squaredistance(a,b):
    # 差の二乗和
    sum = 0.0
    keys = set(a.keys()+b.keys())
    for key in keys:
        sum += (a[key] - b[key]) ** 2
    return sum
#    c = dokdiff(a,b)
#    return sum([ x**2 for x in c.values()])

def distance(a,b):
    # 2-ノルム
    return math.sqrt(squaredistance(a,b))

def mean(d):
    # 平均
    sums = doksum(*d)
    return product_scalar(sums, 1./len(d))

def weighted_choice_bisect_compile(items):
    """Returns a function that makes a weighted random choice from items."""
    added_weights = []
    last_sum = 0

    for item, weight in items:
        last_sum += weight
        added_weights.append(last_sum)

    def choice(rnd=random.random, bis=bisect.bisect):
        return items[bis(added_weights, rnd() * last_sum)][0]
    return choice

def eliminate_zeros(d, eps=1.e-9):
    for k in [ k for k,v in d.iteritems() if abs(v) < eps ]:
        del d[k]
    return d
