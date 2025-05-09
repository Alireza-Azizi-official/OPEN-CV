# برای کشیدن خط باید اول و انتهای خطی که میخوای پاس بدی رو بدی 
import numpy as np
import cv2 as cv

# Create a black image
img = np.zeros((512,512,3), np.uint8)
 
# Draw a diagonal blue line with thickness of 5 px
cv.line(img, (0,0),(511,511),(255,0,0),2)

cv.rectangle(img, (374,10), (490, 128), (0,255,0), 1)

cv.circle(img, (435,70), 32, (0,0,225), -1)

cv.ellipse(img, (150,370), (50,50), 0, 0, 180, 255, -1)

pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
pts = pts.reshape((-1,1,2))
cv.polylines(img,[pts],True,(0,255,255))

font = cv.FONT_HERSHEY_SIMPLEX
cv.putText(img, 'openCV', (10,500), font, 3, (255,255,255), 1, cv.LINE_AA)

cv.imshow('IMage', img)
cv.waitKey(0)
cv.destroyAllWindows()

#------------------------------------------------------------------------------------
# استفاده از موس به عنوان قلمو

import numpy as np
import cv2 as cv

drawing = False
mode = True
ix, iy = 1,1

def draw_circle(event, x, y, flags, param):
    global ix, iy, drawing, mode
    
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                cv.rectangle(img, (ix, iy), (x,y), (0,255,0), 1)
            else:
                cv.circle(img, 5, (0,0,255), 1)
                
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv.rectangle(img, (ix, iy), (x, y), (0,255,0), 1)
        else:
            cv.circle(img, (x,y), 5,(0,0,255), -1)

img = np.zeros((512,512,3),np.uint8)
cv.namedWindow('image')
cv.setMouseCallback('image',draw_circle)

while(1):
    cv.imshow('image', img)
    k = cv.waitKey(1) and 0xFF
    if k == ord('m'):
        mode = not mode
    elif k == 27:
        break
    
cv.destroyAllWindows()

#------------------------------------------------------------------------------------
# ساخت نرم افزار تغییر  رنگ پس زمینه
import numpy as np
import cv2 as cv
 
def nothing(x):
 pass
 
# Create a black image, a window
img = np.zeros((300,512,3), np.uint8)
cv.namedWindow('image')
 
# create trackbars for color change
cv.createTrackbar('R','image',0,255,nothing)
 
cv.createTrackbar('G','image',0,255,nothing)
cv.createTrackbar('B','image',0,255,nothing)
 
# create switch for ON/OFF functionality
switch = '0 : OFF \n1 : ON'
cv.createTrackbar(switch, 'image',0,1,nothing)
 
while(1):
    cv.imshow('image',img)
    k = cv.waitKey(1) & 0xFF
    if k == 27:
        break
    
    r = cv.getTrackbarPos('R','image')
    g = cv.getTrackbarPos('G','image')
    b = cv.getTrackbarPos('B','image')
    s = cv.getTrackbarPos(switch,'image')
    
    if s == 0:
        img[:] = 0
    else:
        img[:] = [b,g,r]
 
cv.destroyAllWindows()