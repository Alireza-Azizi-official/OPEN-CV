import cv2 as cv 

# تابع زیر تعداد دفعاتی که تکرار شده رو بعد از زمان اجرا ارسال میکننه به تابع
# e1 = cv.getTickCount()
# e2 = cv.getTickCount()
# time = (e2 - e1)
# time = (e2 - e1) / cv.getTickFrequency()

import cv2 as cv

e1 = cv.getTickCount()
for i in range(5,49,2):
    img1 = cv.medianBlur(img1, i)
e2 = cv.getTickCount()
t = (e2 - e1)/ cv.getTickFrequency()
print(t)


# to check if optimization is enabled

cv.useOptimized()

res  = cv.medianBlur(img1, 49)
cv.setUseOptimized(False)
cv.useOptimized()
res = cv.medianBlur