# based on http://docs.opencv.org/master/d7/d8b/tutorial_py_lucas_kanade.html and hacked away
import cv2
import numpy as np
import sys

# https://stackoverflow.com/questions/17459584/opencv-warping-image-based-on-calcopticalflowfarneback
def mapFromFlow(flow):
    #print("hackity hack")
    """
    #slow
    mx = np.zeros(flow.shape, np.float32)
    my = np.zeros(flow.shape, np.float32)
    for i in range(0, flow.shape[0]):
        for j in range(0, flow.shape[1]):
            f = flow[i,j]
            mx[i,j] = i + f[0]
            my[i,j] = j + f[1]
    return np.float32(mx), np.float32(my)
    """
    """
    #bad
    mx = np.array([[flow[i, j] + i for j in xrange(flow.shape[1])] for i in xrange(flow.shape[0])])
    my = np.array([[flow[i, j] + j for j in xrange(flow.shape[1])] for i in xrange(flow.shape[0])])
    ## debug
    #cv2.imwrite("map_x.png", mx[..., 0]);
    #cv2.imwrite("map_y.png", my[..., 0]);
    # print mx.shape, mx.dtype

    # something went wrong, we accidentally transposed the map
    mx = mx[..., 0].T.astype('float32')
    my = my[..., 0].T.astype('float32')

    #debug
    # write_map(mx, my)
    """
    # surplus = np.zeros_like(flow)
    # surplus = np.array([[[i, j] for j in xrange(flow.shape[1])] for i in xrange(flow.shape[0])])
    line = np.float32(np.array(range(flow.shape[0])))
    surplusY = np.repeat(line, flow.shape[1]).reshape(flow.shape[0], flow.shape[1])
    surplusX = surplusY.T
    mx = np.float32(flow[..., 0]) + surplusX
    my = np.float32(flow[..., 1]) + surplusY
   
    # print mx.shape, mx.dtype

    return mx, my

def write_map(mx, my):
    print("writing the map for debug purposes")
    zz = np.zeros_like(mx)
    m = np.dstack([mx, my, zz])
    max = np.max(m)
    m /= max
    cv2.imwrite("map.png", m)


def mapFromFlowCBAD(flow):
    m = np.array([[[flow[i, j] + i, flow[i, j] + j] for j in range(flow.shape[1])] for i in xrange(flow.shape[0])])
    # debug
    cv2.imwrite("map_x.png", m[...,0]);
    cv2.imwrite("map_y.png", m[...,1]);
    m.reshape(flow.shape, dtype=np.complex128)
    return m

def mapFromFlowC(flow):
    mx, my = mapFromFlow(flow)
    m = 1j * my # convert to complex
    m += mx
    m = m.view(dtype=np.complex64)[...,0]
    return m

def mapFromFlowD(flow):
    mx, my = mapFromFlow(flow)
    m = np.dstack([mx, my])
    # debug 
    write_map(mx, my)
    return m



# hack: read in color for color hsv dimensions, then read greyscale
name = sys.argv[1] 
print("reading %s" % name)
prev = cv2.imread(name, 1)

hsv = np.zeros(prev.shape, prev.dtype)
hsv[...,1] = 255

print("reading %s" % name)
prev = cv2.imread(name, 0)

name = sys.argv[2]
print("reading %s" % name)
next_ = cv2.imread(name, 0)


print("calculating flow for ", name)
flow = cv2.calcOpticalFlowFarneback(next_, prev, None, 0.5, 3, 15, 3, 5, 1.2, 0)
# https://stackoverflow.com/questions/17459584/opencv-warping-image-based-on-calcopticalflowfarneback

"""
print("flow done, interpolating...")
num_steps = int(sys.argv[4]) + 1
for step in range(1, num_steps+1):
    distmx, distmy = mapFromFlow((float(step) / float(num_steps)) * flow)
    # distm = mapFromFlowD(flow*step/num_steps)
    newf = np.zeros(prev.shape, prev.dtype)
    print("remapping ", step, " of ", num_steps)
    newf = cv2.remap(prev, distmx, distmy, cv2.INTER_CUBIC)
    # newf = cv2.remap(prev, distm, None, cv2.INTER_CUBIC)
    name = "xx_interpol_hin_%02d_%d.png" % (i, step)
    print("writing interpolated file ", name, "...")
    cv2.imwrite(name, newf)


print("writing orig")
name = "xx_interpol_hin_%02d_%d.png" % (i, 0)
cv2.imwrite(name, prev)
#name = "xx_interpol_hin_%02d_%d.png" % (i, step+1)
#cv2.imwrite(name, next_)
"""

print("beautifying...")
## we gave hsv in right dimensions from color loading
mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
# print mag.shape
# print ang.shape
hsv[...,0] = ang * 180 / np.pi / 2
hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    
print("writing flow file ", sys.argv[3])
cv2.imwrite(sys.argv[3], bgr)
# prev = next

