import cv2
import numpy as np
import random as r
import sys
import scipy
import codecs, json
import os.path
import pickle

class Params:
    has_add_border_before = True
    add_border_factor = 4
    sigma_prescale_factor = 10.0
    sigma_n = 10
    write_distortion = True
    distortion_as_json = False
    write_visualization = True
    do_rigid = True
    rigid_downscale = 4.0
    has_special_scale = False
    special_scale = None
    gauss_radius_factor = 4
    has_premature_distort = True
    premature_distort_factor = 2.0
    has_final_gauss = True
    final_gauss_radius = 11
    background_value = 255
    remap_type = cv2.INTER_CUBIC

def base_map(shape):
    line = np.float32(np.array(range(shape[0])))
    surplusY = np.repeat(line, shape[1]).reshape(shape[0], shape[1])
    surplusX = surplusY.T
    return surplusX, surplusY

def convert_map(flowX, flowY):
    line = np.float32(np.array(range(flowX.shape[0])))
    surplusY = np.repeat(line, flowX.shape[1]).reshape(flowX.shape[0], flowX.shape[1])
    surplusX = surplusY.T
    mx = np.float32(flowX) + surplusX
    my = np.float32(flowY) + surplusY
    return mx, my


def displacements_scale(sigma, scale, n, img, mu=0.0):
    """
    print("generating randoms for n=%d..." % n)
    offset_loc_x = normal(mu, sigma, (n,n))
    offset_loc_y = normal(mu, sigma, (n,n))
    """
    
    shape = (int(img.shape[0] / 2**scale), int(img.shape[1] / 2**scale))
    print("creating displacements at", shape)
    distort_x = np.zeros(shape[0:2], dtype=np.float)
    distort_y = np.zeros_like(distort_x)
    """
    print(np.max(offset_loc_x))
    print(np.max(offset_loc_y))
    """
    
    for i in range(0,n):
        for j in range (0,n):
            u = int(i * float(shape[0]) / n)
            v = int(j * float(shape[1]) / n)
            # print("u=%d, v=%d, scale=%d, offsetx=%f, offsety=%f" % (u,v, scale, offset_loc_x[i, j], offset_loc_y[i, j]))        
            distort_x[u,v] = r.normalvariate(mu, sigma)
            distort_y[u,v] = r.normalvariate(mu, sigma)

            # print("i=%d, j=%d, u=%d, v=%d, dx=%f, dy=%f" % (i,j,u,v,distort_x[u,v], distort_y[u,v]));

    # print(distort_y.shape, distort_y)

    # print("resizing displacements to", img.shape[0:2])
    sx = cv2.resize(distort_x, img.shape[0:2])
    sy = cv2.resize(distort_y, img.shape[0:2])

    gauss_r = 4 * scale + 1
    rx = cv2.GaussianBlur(sx, (gauss_r, gauss_r), 0)
    ry = cv2.GaussianBlur(sy, (gauss_r, gauss_r), 0)
    # print(rx)

    return rx, ry

print("disorter called with\n- parameter file",sys.argv[1],
      "\n- input image", sys.argv[2],
      "\n- locally distorted file", sys.argv[3],
      "\n- globally distorted file", sys.argv[4],
      "\n- visualization file", sys.argv[5], "\n")

filename = sys.argv[1]
if os.path.isfile(filename):
    print("loading parameter file:", filename)
    with open(filename, 'rb') as file:
        params = pickle.load(file)
else:
    # parameter file not found, let's create it
    params = Params()
    print("SAVING parameter file:", filename)
    with open(filename, 'wb') as file:
        pickle.dump(params, file)

name = sys.argv[2]
print("reading %s..." % name)
im = cv2.imread(name, -1)

if params.has_add_border_before:
    ## pad the image beforehand
    add_border = int(im.shape[0]/params.add_border_factor)
    bg_val = tuple([int(params.background_value) for x in [1,2,3]])
    ext = cv2.copyMakeBorder(im, add_border, add_border, add_border, add_border, cv2.BORDER_CONSTANT, value=bg_val)
    cv2.imwrite(sys.argv[2] + ".dump.png", ext)
    # ext = import
else:
    ext = im

side = max(im.shape[0], im.shape[1])
if not params.has_special_scale:
    scales = np.log2(side)
else:
    scales = params.special_scale
sigmas = [ params.sigma_prescale_factor/float(i) for i in range(1, int(scales))]
print("sigma stack is:", sigmas)
side = None

if params.has_premature_distort:
    tx, ty = displacements_scale(sigmas[0], int(scales)-1, 2, ext)
    acc_x = tx*2.0
    acc_y = ty*2.0
else:
    acc_x, acc_y = base_map(ext.shape)

for i in range(int(scales)-2, 0, -2):
    print("range %d..." % i)
    cx, cy = displacements_scale(sigmas[i], i, params.sigma_n, ext)
    acc_x += cx
    acc_y += cy

if params.has_final_gauss:
    print("gauss again...")
    acc_x = cv2.GaussianBlur(acc_x,
       (params.final_gauss_radius, params.final_gauss_radius), 0)
    acc_y = cv2.GaussianBlur(acc_y,
       (params.final_gauss_radius, params.final_gauss_radius), 0)

    
print("converting maps...")
mx, my = convert_map(acc_x, acc_y)
# print(mx.shape, mx)
# print(my.shape, my)

if params.write_distortion:
    print("dumping maps...")
    if params.distortion_as_json:
        name = sys.argv[3] + "-local.json"
        # https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
        json.dump((mx.tolist(), my.tolist()), codecs.open(name, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=False, indent=4)
        ### this saves the array in .json format
    else:
        name = sys.argv[3] + "-local.png"
        cx = np.array(acc_x, dtype=float)
        cy = np.array(acc_y, dtype=float)
        dump_img = np.zeros((ext.shape[0], ext.shape[1], 3), dtype=np.float)
        dump_img[...,0] = cx
        dump_img[...,1] = cy
        print("writing local distortion dump %s..." % name)
        cv2.imwrite(name, dump_img)
        dump_img = None

print("remapping...")
bg_val = tuple([int(params.background_value) for x in [1,2,3]])
newf = cv2.remap(ext, mx, my, params.remap_type, cv2.BORDER_CONSTANT, borderValue=bg_val)

name = sys.argv[3]
print("writing result file ", name, "...")
cv2.imwrite(name, newf)

if len(sys.argv) > 4 and params.write_visualization:
    print("beautifying...")
    mag, ang = cv2.cartToPolar(cx, cy)
    hsv = np.zeros((ext.shape[0], ext.shape[1], 3), dtype=np.uint8)
    ### for color: fill ch.1 white
    hsv[...,1].fill(255)
    # hsv[...,0] = ang * 180 / np.pi / 2
    ## careful: normalised in each transformation on its own!
    hsv[...,0] = cv2.normalize(ang * 180 / np.pi / 2,None,0,255,cv2.NORM_MINMAX)
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

    name = sys.argv[5]
    print("writing local distortion visualization %s..." % name)
    cv2.imwrite(name, bgr)

if params.do_rigid:
    print("global transform...")
    angle = r.uniform(-180.0, 180.0)
    downscale = params.rigid_downscale
    xshift = r.uniform(-im.shape[0]/downscale, im.shape[0]/downscale)
    yshift = r.uniform(-im.shape[1]/downscale, im.shape[1]/downscale)
    xcenter = newf.shape[0]/2.0
    ycenter = newf.shape[1]/2.0
    print("shifting %d, %d" % (xshift, yshift))
    rotmat = cv2.getRotationMatrix2D((xcenter, ycenter), angle, 1.0)
    transmat = np.float32([[1,0,xshift], [0,1,yshift]])
    print("rotmat\n", rotmat)
    json.dump((rotmat.tolist(), xshift, yshift), codecs.open(sys.argv[4] + ".global.json", 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
    
    xshift = abs(int(xshift))
    yshift = abs(int(yshift))
    bg_val = tuple([int(params.background_value) for x in [1,2,3]])
    # res = cv2.copyMakeBorder(newf, xshift, yshift, xshift, yshift, cv2.BORDER_CONSTANT, value=0)
    res = cv2.warpAffine(newf, transmat, ext.shape[0:2], cv2.BORDER_CONSTANT, borderValue=bg_val)
    final = cv2.warpAffine(res, rotmat, ext.shape[0:2], cv2.BORDER_CONSTANT, borderValue=bg_val)
    name = sys.argv[4]
    print("writing globally distorted file %s..." % name)
    cv2.imwrite(name, final)
