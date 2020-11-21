## initial code by Jean Francois Pambrun
## https://gist.github.com/jpambrun/0cf01ab512db036ae32a4834a4cd542e
import cv2
import numpy as np
import sys

def load(name, thr=100, is_otsu=True, is_blur_otsu=True):
    print("loading", name)
    img = cv2.imread(name, 0)

    if is_otsu:
        print("otsu...")
        # https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html
        # Otsu's thresholding after Gaussian filtering
        if is_blur_otsu:
            blur = cv2.GaussianBlur(img,(5,5),0)
            if len(sys.argv) > 7:
                cv2.imwrite(sys.argv[7], blur)
        else:
            blur = img
        ret3,th = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    else:
        ret3,th = cv2.threshold(img,thr,255,cv2.THRESH_BINARY)

    return img, th

img1, th1 = load(sys.argv[1])
img2, th2 = load(sys.argv[2])

print("diff...")
diff = ((th1.astype(np.int16) - th2.astype(np.int16))/2+128).astype(np.uint8)
im_color = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
lut = np.zeros((256,1,3), dtype = "uint8")
for i in range(256):
    # print(i)
    lut[i,0,0] = max(min((127-i)*2,255),0) if i < 127 else 0
    lut[i,0,1] = max(min((i-127)*2,255),0) if i > 127 else 0
    lut[i,0,2] = max(min((127-i)*2,255),0) if i < 127 else 0

im_falsecolor_r = cv2.LUT(diff, lut[:,0,0])
im_falsecolor_g = cv2.LUT(diff, lut[:,0,1])
im_falsecolor_b = cv2.LUT(diff, lut[:,0,2])
im_falsecolor = np.dstack((im_falsecolor_r, im_falsecolor_g, im_falsecolor_b))

## new:
print("mult...")
both = cv2.cvtColor(cv2.multiply(th1, th2), cv2.COLOR_GRAY2RGB)
# ^^^ what both have in common, in white; then convert to RGB
# print(im_falsecolor.shape, both.shape)
final = cv2.add(im_falsecolor, both)

print("saving", sys.argv[3])
cv2.imwrite(sys.argv[3], final)

if len(sys.argv) > 4:
    print("saving dumps:", sys.argv[4])
    cv2.imwrite(sys.argv[4], th1)
    print("saving dumps:", sys.argv[5])
    cv2.imwrite(sys.argv[5], th2)
    print("saving diff-only:", sys.argv[6])
    cv2.imwrite(sys.argv[6], im_falsecolor)
    
