import cv2 as cv
import numpy as np 
from matplotlib import pyplot as plt
import math as ma
imgL = cv.imread('prac_L.JPEG')
imgL1=cv.cvtColor(imgL,cv.COLOR_BGR2GRAY)

imgR = cv.imread('prac_R.JPEG')
imgR1=cv.cvtColor(imgR,cv.COLOR_BGR2GRAY)

# Initiate SIFT detector
sift = cv.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(imgL1,None)
kp2, des2 = sift.detectAndCompute(imgR1,None)

# BFMatcher with default params
bf = cv.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)
# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append(m)

src_p = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
dst_p=np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

M,msk=cv.findHomography(src_p,dst_p,cv.RANSAC,5.0)
matchMsk=msk.ravel().tolist()

x,y=imgL1.shape
m,n=imgR1.shape
#trans_pts=np.float32([[0,0],[0,x-1],[x,y],[y-1,0]]).reshape(-1,1,2)
#dst=cv.perspectiveTransform(trans_pts,M)

warp = cv.warpPerspective(imgR,M,(imgL.shape[1] + imgR.shape[1], imgR.shape[0]+imgL.shape[0]))
plt.imshow(warp)
plt.show()

'''val=dst[0:imgL.shape[0], 0:imgL.shape[1]]
val=val.copy()
plt.imshow(val)
plt.show()'''
warp[0:imgL.shape[0], 0:imgL.shape[1]] = imgL

#dst[0:imgL.shape[0] ,imgL.shape[1]:dst.shape[1] ]=val
plt.imshow(warp)
plt.show()


'''st_img=cv.warpPerspective(imgL,M,(n+y,x))
a,b=st_img.shape'''

'''for i in range(a):
	for j in range(b):
		if 6'''




# cv.drawMatchesKnn expects list of lists as matches.
#img3 = cv.drawMatchesKnn(imgL,kp1,imgR,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#plot.imshow(st_img),plot.show()
#plot.imshow(imgL),plot.show()
