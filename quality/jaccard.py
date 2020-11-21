# based on http://docs.opencv.org/master/d7/d8b/tutorial_py_lucas_kanade.html and hacked away
from skimage import io
import sklearn.metrics as m
import numpy as np
import sys

t = int(sys.argv[3])
print("threshold:", t)

name = sys.argv[1]
print("reading %s" % name)
prev_ = io.imread(name)
p = prev_.reshape(-1, 1).flatten() > t

name = sys.argv[2]
print("reading %s" % name)
next_ = io.imread(name)
n = next_.reshape(-1, 1).flatten() > t

score = m.jaccard_score(p, n)
print("Jaccard: ", score)

