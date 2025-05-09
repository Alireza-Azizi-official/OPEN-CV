# حذف نویز های کوچک و کم از تصویر 
"""
نحوه ی عملکرد این الگوریتم به این صورت هست که از گوشه شروع میکنه تا وسط تمام پیکسل ها رو چک میکنه 
بعد قسمت هایی که مشکل دارند رو انتخاب میکنه و با استفاده از پیکسل های اطرافش پیکسل های خراب رو پر میکنه
و در نهایت ترمیم میکنه و عکس ترمیم شده رو میده بهمون.
"""

"""
مراحل انجام کار:
اول یک ماسک از عکس میسازیم
"""

# import numpy as np 
# import cv2 as cv

# img = cv.imread(r'code\noizy.jpg')
# mask = cv.imread('mask2.png', cv.IMREAD_GRAYSCALE)
# dst = cv.inpaint(img, mask, 3, cv.INPAINT_TELEA)

# cv.imshow('dst', dst)
# cv.waitKey(0)
# cv.destroyAllWindows()

# تغییر نور عکس 
