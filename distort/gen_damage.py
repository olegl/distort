import numpy as np
import random as r
import sys
import scipy

print("# generated damaging script...")
w = int(sys.argv[1])
h = int(sys.argv[2])
fill = sys.argv[3]
input = sys.argv[4]
output = sys.argv[5]

side = max(w, h)
# print("convert -size", "%dx%d" % (w,h) ," xc:none -colorspace Gray -background none ", output)
print("convert ", input," -background none ", output)
for i in range(0,10):
    x = int(r.normalvariate(side/2, side/2))
    y = int(r.normalvariate(side/2, side/2))
    a = int(r.normalvariate(10, side/10))
    b = int(r.normalvariate(10, side/10))
    #angle = int(r.normalvariate(18, 18))*10 # bin the angles
    # e_angle = abs(int(r.normalvariate(180, 180)))
    # if s_angle > e_angle:
    #    (s_angle, e_angle) = (e_angle, s_angle)
    temp = "temp_%02d.png" % i
    print("convert ",output," \( -fill ",fill," -stroke ",fill, " \\")
    print(" -draw ' rectangle ", x,",",y," ",x+a,",",x+b, " '  \) -flatten ", temp, "\\")
    print("  && mv ", temp, output)
