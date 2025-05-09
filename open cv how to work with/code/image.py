
# # how to work with images

# import cv2 as cv
# import sys

# img = cv.imread(cv.samples.findFile(r"open cv how to work with\800px-Watt,_James_(Van_Schendel,_1869).jpg"))

# if img is None:
#     sys.exit("there is no such file")
    
# cv.imshow("test", img)
# a = cv.waitKey(0)

# # to save the image
# if img == ord('s'):
#     cv.imwrite("test", img)

# #-----------------------------------------------------------------------
# # دسترسی به پیکسل عکس ها 
# #دسترسی به مشخصه های یک عکس
# #تنظیم ROI
# # جدا کردن و چسباندن عکس ها 

# import numpy as np
# import cv2 as cv

# img = cv.imread(cv.samples.findFile(r"open cv how to work with\800px-Watt,_James_(Van_Schendel,_1869).jpg"))


# a = img.item(10,10,2)
# img.itemset((10,10,2),100)
# b = img.item(10,10,2)
# # print(a)


# #دسترسی به ویژگی های عکس 
# print(img.shape)

# #اگر خواستی سایز کلی عکس رو بدست بیاری زیر و بزنی 
# print(img.size )

# #گرفتن دیتا تایپ یک عکس 
# #این خیلی مهمه چون بیشتر اررو ها در اوپن سی وی برای دادن ورودی با تایپ نادرست هست
# print(img.dtype)

# #-------------------------------------------------------------------------
# # image ROI
# # این کار به ما کمک میکنه تا قسمت مشخصی رو انتخاب کنیم و دنبال چیزی که میخوایم بگردیم
# import numpy as np
# import cv2 as cv

# # #با این روش میتونیم یک ایتم در یک عکس رو در قسمت دیگه ای از عکس قرار بدیم 
# img2 = cv.imread(r'open cv how to work with\test.jpg')
# head = img2[230:340, 330:390]
# img2[123:453, 100:160] = head
# #-------------------------------------------------------------------------
# #جدا کردن و چسباندن عکس ها به یکدیگر 
# import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt 

# img = cv.imread(r'open cv how to wrok with\test.jpg')
# b, g, r = cv.split(img)
# img = cv.merge((b,g,r))

# #split زمان بره میتونی از نامپای استفاده کنی 

# # برای انداختن فریم دور عکس 

# BLUE = [255, 0, 0]
# img = cv.imread(r'open cv how to wrok with\test.jpg')

# replicate = cv.copyMakeBorder(img, 10, 10, 10, 10, cv.BORDER_REPLICATE)
# reflect = cv.copyMakeBorder(img, 10, 10 ,10, 10, cv.BORDER_REFLECT)
# reflect101 = cv.copyMakeBorder(img, 10, 10, 10, 10, cv.BORDER_REFLECT_101)
# wrap = cv.copyMakeBorder(img, 10, 10, 10,10,cv.BORDER_WRAP)
# constant = cv.copyMakeBorder(img, 10, 10, 10, 10, cv.BORDER_CONSTANT, value=BLUE)

# plt.subplot(231),plt.imshow(img,'gray'),plt.title('original')
# plt.subplot(232),plt.imshow(replicate,'gray'),plt.title('replicate')
# plt.subplot(233),plt.imshow(reflect,'gray'),plt.title('reflect')
# plt.subplot(234),plt.imshow(reflect101,'gray'),plt.title('reflect101')
# plt.subplot(235),plt.imshow(wrap,'gray'),plt.title('wrap')
# plt.subplot(236),plt.imshow(constant,'gray'),plt.title('constant')
# plt.show()

# #---------------------------------------------------------------------------
# #arithmetioc operations on images
# #image addition 
# #برای اضافه کردن عکس دو عکس باید تماما عمق یکسانی داشته باشند یا عکس دومی باید بالاتر باشد

# import numpy as np
# import cv2 as cv
# import matplotlib.pyplot as plt 

# x = np.uint8([250])
# y = np.uint8([10])
# print(cv.add(x,y))
# print(x+y)



# # #روش زیر دو تا عکس رو باهم ترکیب میکنه منتهی از هرکدوم با یه مقدار تاری میده و میندازه رو هم
# img1 = cv.imread(r'open cv how to wrok with\test.jpg')
# img2 = cv.imread(r'open cv how to wrok with\test.jpg')

# dst = cv.addWeighted(img1,0.7, img2,0.3, 0)
# cv.imshow('dst', dst)
# cv.waitKey(0)
# cv.destroyAllWindows()


# import numpy as np
# import cv2 as cv
# import matplotlib.pyplot as plt

# img1 = cv.imread('test.jpg')
# img2 = cv.imread('test.jpg')

# rows1, cols1, channels1 = img1.shape
# rows2, cols2, channels2 = img2.shape

# roi = img1[0:rows1, 0:cols1]

# img2gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
# ret, mask = cv.threshold(img2gray, 10, 255, cv.THRESH_BINARY)
# mask_inv = cv.bitwise_not(mask)

# # Now black-out the area of logo in ROI
# img1_bg = cv.bitwise_and(roi, roi, mask = mask_inv)

# # Take only region of logo from logo image.
# img2_fg = cv.bitwise_and(img2, img2, mask = mask)

# # Put logo in ROI and modify the main image
# dst = cv.add(img1_bg, img2_fg)
# img1[0:rows1, 0:cols1 ] = dst

# cv.imshow('res', img1)
# cv.waitKey(0)
# cv.destroyAllWindows()
#----------------------------------------------------------------------------------
#برای تغییر سایز یک عکس میتونیم به روش زیر استفاده کنیم 

# import numpy as np
# import cv2 as cv 
# import matplotlib.pyplot as plt 

# img = cv.imread(cv.samples.findFile(r'test.jpg'))
# res = cv.resize(img,fx=2 ,fy=2, interpolation=cv.INTER_CUBIC)

#اگر که شماره ارایه جای مشخصی از یک ارایه رو داشته باشی میتونی میتونی جاشو عوض کنی 

# img = cv.imread(r'D:\9-PYTHON COURSE\PYTHON\open cv how to work with\test.jpg',cv.IMREAD_GRAYSCALE)
# rows, cols = img.shape

# M = np.float32([[1,0,100],[0,1,50]])
# dst = cv.warpAffine(img, M, (cols,rows))

# cv.imshow('img', dst)
# cv.waitKey(0)
# cv.destroyAllWindows()


#rotate 
# img = cv.imread(r'D:\9-PYTHON COURSE\PYTHON\open cv how to work with\test.jpg')rows, cols = img.shape
# M = cv.getRotationMatrix2D(((cols-1)/ 2.0, (rows-1)/2.0),90,1)
# dst = cv.warpAffine(img, M,(cols,rows))

# img = cv.imread(r'D:\9-PYTHON COURSE\PYTHON\open cv how to work with\test.jpg')
# rows, cols, ch = img.shape

# pts1 = np.float32([[50,50], [200,50], [50,200]])
# pts2 = np.float32([[10,100],[200,50], [100,250]])

# M = cv.getAffineTransform(pts1, pts2)
# dst = cv.warpAffine(img, M,(cols,rows))

# plt.subplot(121), plt.imshow(img), plt.title('Input')
# plt.subplot(122), plt.imshow(dst), plt.title('Output')
# plt.show()

#برای اینکه بتونیم عکسی بگیریم که حاشیه عکس رو بگیره و فقط مربع درخواستی رو بده به صورت زیر عمل میکنیم
#برای این کار نیاز به یک ماتریکس سه در سه داریم
# img = cv.imread(r'D:\9-PYTHON COURSE\PYTHON\open cv how to work with\test.jpg')
# row, cols, ch = img.shape

# pts1 = np.float32([[56,65],[368,52], [28, 387], [389,390]])
# pts2 = np.float32([[0,0], [300,0],[0,300], [300,300]])

# M = cv.getPerspectiveTransform(pts1, pts2)
# dst = cv.warpPerspective(img, M, (300,300))

# plt.subplot(121), plt.imshow(img), plt.title('Input')
# plt.subplot(121), plt.imshow(dst), plt.title('Output')
# plt.show()

#---------------------------------------------------------------------------------
#برای گرفتن نویز عکس های متفاوت و گرفتن و یا دادن سایه به عکس به صورت زیر 
# img = cv.imread(r'D:\9-PYTHON COURSE\PYTHON\open cv how to work with\test.jpg', cv.IMREAD_GRAYSCALE)
# img = cv.medianBlur(img, 5)
# ret, th1 = cv.threshold(img, 127,255, cv.THRESH_BINARY)
# th2 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
# th3 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
# titles = ['Original Image', 'Global Thresholing (v = 127)', 'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
# images = [img, th1, th2, th3]
# for i in range(4):
#     plt.subplot(2,2,i+1), plt.imshow(images[i], 'gray')
#     plt.title(titles[i])
#     plt.xticks([]), plt.yticks([])
# plt.show()
#-----------------------------------------------------------------------------------
# import cv2 as cv
# import numpy as np
# from matplotlib import pyplot as plt 


# img = cv.imread(r'D:\9-PYTHON COURSE\PYTHON\open cv how to work with\test.jpg', cv.IMREAD_GRAYSCALE)
# ret1, th1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
# ret2, th2 = cv.threshold(img, 0,255, cv.THRESH_OTSU)

# blur = cv.GaussianBlur(img, (5,5), 0)
# ret3, th3 = cv.threshold(blur, 0, 255, cv.THRESH_OTSU)

# images = [img, 0, th1, img, 0, th2, blur, 0, th3]
# titles = ['Original Noisy Image', 'Histogram', 'Global Thresholding (v = 127)', 'Original Noisy Image', 'Histogram', 'Otsus threshholding', 'Gaussian filtered Image','Histogram', 'Otsus Thresholding']

# for i in range(3):
    
#     plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
    
#     plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
    
#     plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
    
#     plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
    
#     plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
    
#     plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
    
# plt.show()

#-----------------------------------------------------------------------------------
# blur images 
# custom filters


# import numpy as np 
# import cv2 as cv 
# from matplotlib import pyplot as plt 

# img = cv.imread(r'D:\9-PYTHON COURSE\PYTHON\open cv how to work with\test.jpg')
# kernel = np.ones((5,5), np.float32) / 25
# dst = cv.filter2D(img, -1, kernel)

# plt.subplot(121), plt.imshow(img), plt.title('Original')
# plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(dst), plt.title('Averging')
# plt.xticks([]), plt.yticks([])
# plt.show()

#مدل 2
# img = cv.imread(r'D:\9-PYTHON COURSE\PYTHON\open cv how to work with\test.jpg', cv.IMREAD_GRAYSCALE)
# blur = cv.GaussianBlur(img,(5,5),0)    
# blur = cv.blur(img,(5,5))
# blur = cv.medianBlur(img, 5)
# blur = cv.bilateralFilter(img,9, 75, 75)

# plt.subplot(121),plt.imshow(img),plt.title('Original')
# plt.xticks([]),plt.yticks([])
# plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
# plt.xticks([]),plt.yticks([])
# plt.show()
#-----------------------------------------------------------------------------------
# morphological transformations 
# افکت دادن به عکس مثلا یک متنی رو دون دون کنی یک متنی رو تو خالی کنی و اینجور چیزا


# img = cv.imread(r'D:\9-PYTHON COURSE\PYTHON\open cv how to work with\test2.jpg')
# kernel = np.ones((5,5), np.uint8)
# erosion = cv.erode(img, kernel, iterations = 1)
# dilation = cv.dilate(img, kernel, iterations= 1)
# opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
# closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
# gradient = cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel)
# tophat = cv.morphologyEx(img,cv.MORPH_TOPHAT, kernel)
# blackhat = cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel)
# cv.imshow('test', blackhat)
# cv.waitKey(0)
#-----------------------------------------------------------------------------------
#برای پیدا کردن عناصر یک عکس و لبه های عکس
# import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt 

# img = cv.imread(r'D:\9-PYTHON COURSE\PYTHON\open cv how to work with\box.jpg',cv.IMREAD_GRAYSCALE)
# laplacian = cv.Laplacian(img, cv.CV_64F)
# sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)
# sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=5)

# plt.subplot(2, 2, 1),plt.imshow(img, cmap='gray')
# plt.title('Original'), plt.xticks([]), plt.yticks([])
# plt.subplot(2, 2, 2), plt.imshow(laplacian, cmap='gray'), plt.yticks([])
# plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
# plt.subplot(2, 2, 3), plt.imshow(sobelx, cmap='gray')
# plt.title('sobel x'), plt.xticks([]), plt.yticks([])
# plt.subplot(2, 2, 4), plt.imshow(sobely, cmap='gray')
# plt.title('sobel y'), plt.xticks([]), plt.yticks([])
# plt.show()


# sobelx = cv.Sobel(img, cv.CV_8U, 1, 0, ksize=5)
# sobelx64f = cv.Sobel(img,cv.CV_64F,1,0,ksize=5)
# abs_sobel64f = np.absolute(sobelx64f)
# sobel_8u = np.uint8(abs_sobel64f)

# plt.subplot(1, 3, 1,),plt.imshow(img, cmap='gray')
# plt.title('Original'), plt.xticks([]), plt.yticks([])
# plt.subplot(1, 3, 2), plt.imshow(sobel_8u, cmap='gray')
# plt.title('Sobel CV_8U'), plt.xticks([]), plt.yticks([])
# plt.subplot(1, 3, 3), plt.imshow(sobel_8u, cmap= 'gray')
# plt.title('Sobel abs(cv_64f)'), plt.xticks([]), plt.yticks([])
# plt.show()

#-----------------------------------------------------------------------------------
#canny edge detection 
#پیدا کردن لبه های یک عکس 

# import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt 

# img = cv.imread(r'open cv how to work with\messi.jpg', cv.IMREAD_GRAYSCALE)

# if img is None:
#     print('file could not be read, check with os.path.exists')
    
# edges = cv.Canny(img, 100, 200)
# # cv.imshow('test',edges)
# # cv.waitKey(0)
# plt.subplot(121), plt.imshow(img, cmap='gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(121), plt.imshow(edges, cmap='gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
# plt.show()

#-----------------------------------------------------------------------------------
# image pyramids 
# برای ترکیب دو عکس به صورت یک عکس و ساخت هرم عکسی 

# img = cv.imread(r'open cv how to work with\messi.jpg')
# lower_reso = cv.pyrDown(higher_reso)
# higher_reso2 = cv.pyrUp(lower_reso)


# import cv2 as cv
# import numpy as np 

# a = cv.imread(r'D:\9-PYTHON COURSE\PYTHON\open cv how to work with\apple.jpg')
# b = cv.imread(r'D:\9-PYTHON COURSE\PYTHON\open cv how to work with\orange.jpg')

# G = a.copy()
# gpA = [G]
# for i in range(6):
#     G = cv.pyrDown(G)
#     gpA.append(G)
 
# G = b.copy()
# gpB = [G]
# for i in range(6):
#     G = cv.pyrDown(G)
#     gpB.append(G)
 
# lpA = [gpA[5]]
# for i in range(5,0,-1):
#     GE = cv.pyrUp(gpA[i])
#     GE = cv.resize(GE, (gpA[i-1].shape[1], gpA[i-1].shape[0]))
#     L = cv.subtract(gpA[i-1],GE)
#     lpA.append(L)
 
# lpB = [gpB[5]]
# for i in range(5,0,-1):
#     GE = cv.pyrUp(gpB[i])
#     GE = cv.resize(GE, (gpB[i-1].shape[1], gpB[i-1].shape[0]))
#     L = cv.subtract(gpB[i-1],GE)
#     lpB.append(L)
    
# LS = []
# for la,lb in zip(lpA,lpB):
#     rows,cols,dpt = la.shape
#     la_cols = cols // 2
#     lb_cols = cols - la_cols
#     min_rows = min(la.shape[0], lb.shape[0])
#     la_cropped = la[:min_rows, :, :]
#     lb_cropped = lb[:min_rows, :la_cols, :]
    
#     ls = np.hstack((la_cropped[:, :la_cols], lb_cropped))
#     LS.append(ls)
 
# ls_ = LS[0]
# for i in range(1, 6):
#     ls_resize = cv.resize(ls_, (LS[i].shape[1], LS[i].shape[0]))
#     ls_ = cv.add(ls_resize, LS[i])
    
# rows, cols, _ = a.shape
# min_rows = min(a.shape[0], b.shape[0])
# a_cropped = a[:min_rows, :, :]
# b_cropped = b[:min_rows, :, :]

# real = np.hstack((a_cropped[:,:cols//2], b_cropped[:,cols//2:]))
 
 
# cv.imwrite('Pyramid_blending2.jpg',ls_)
# cv.imwrite('Direct_blending.jpg',real)

#-----------------------------------------------------------------------------------
# کشیدن خط 

# import numpy as np
# import cv2 as cv
# from matplotlib  import pyplot as plt 

# img = cv.imread(r'open cv how to work with\white.jpg')
# imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# ret, thresh = cv.threshold(imgray, 127, 255, 0)
# contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

# if len(contours) > 0:
#     cv.drawContours(img, contours, -1, (0,255,0), 2)

#     for i, cnt in enumerate(contours):
#         cv.drawContours(img, [cnt], 0, (0,255,0), 2)

# cv.imshow('Image', img)
# cv.waitKey(0)
# cv.destroyAllWindows()
#-----------------------------------------------------------------------------------
#انداختن خط در دور تا دور یک عکس با پستی و بلندی 
# img = cv.imread(r'D:\9-PYTHON COURSE\PYTHON\open cv how to work with\star.jpg', cv.IMREAD_GRAYSCALE)
# ret,thresh = cv.threshold(img,127,255,0)
# contours,hierarchy = cv.findContours(thresh, 1, 2)
# cnt = contours[0]
# # M = cv.moments(cnt)
# (x,y),radius = cv.minEnclosingCircle(cnt)
# center = (int(x),int(y))
# radius = int(radius)
# ellipse = cv.fitEllipse(cnt)
# rows,cols = img.shape[:2]
# [vx,vy,x,y] = cv.fitLine(cnt, cv.DIST_L2,0,0.01,0.01)
# lefty = int((-x*vy/vx) + y)
# righty = int(((cols-x)*vy/vx)+y)
# cv.line(img,(cols-1,righty),(0,lefty),(0,255,0),2)

# cv.ellipse(img,ellipse,(0,255,0),2)
# cv.circle(img,center,radius,(0,255,0),2)

# cv.imshow('test',img)
# cv.waitKey(0)
# cv.destroyAllWindows()
#-----------------------------------------------------------------------------------
#ساخت هیستوگرام برای یک عکس 
#image = عکسی که میدی / channel = اگه عکس گری بود مقدار صفر و اگه رنگی بود یک دو میدیم
#mask = اگه خواستیم کل عکس رو پیدا کنیم مقدار رو نان میدیم اکه نه که یه ماسک درست میکنیم و بعد اونو میگیریم
#histsize =  عدد 256 میگیره
#ranges = رنج هست که معمولا بین 0 تا 256 هست 

# img = cv.imread(r'D:\9-PYTHON COURSE\PYTHON\open cv how to work with\apple.jpg', cv.IMREAD_GRAYSCALE)
# if img is None:
#     print('file couldnt be read')
# # hist = cv.calcHist([img], [0], None, [256], [0,256])
# #or
# hist, bins = np.histogram(img.ravel(), 256, [0,256])
# plt.hist(img.ravel(), 256, [0,256])
# plt.show()

# اگر خواستی قسمتی از عکس رو پیدا کننی یک عکس سفید میندازی رو کل عکس مشکی شده و میندازی روی عکس اصلی و بعد بهت میده ماسکشو
# img = cv.imread(r'D:\9-PYTHON COURSE\PYTHON\open cv how to work with\original.jpg', cv.IMREAD_GRAYSCALE)

# mask = np.zeros(img.shape[:2], np.uint8)
# mask[100:300 , 100:400] = 25
# masked_img = cv.bitwise_and(img, img, mask= mask)

# hist_full = cv.calcHist([img], [0], None, [256], [0,256])
# hist_mask = cv.calcHist([img], [0], mask, [256], [0,256])

# plt.subplot(221), plt.imsave(img, 'gray')
# plt.subplot(222), plt.imshow(mask, 'gray')
# plt.subplot(223), plt.imshow(masked_img, 'gray')
# plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask)
# plt.show()

#برای وضوح بیشتر تصویر به صورت زیر میزنیم
# img = cv.imread(r'open cv how to work with\orig.jpg',cv.IMREAD_GRAYSCALE)
# equ = cv.equalizeHist(img)
# res = np.hstack((img, equ))
# cv.imshow('test', res)
# cv.waitKey(0)

# img = cv.imread(r'D:\9-PYTHON COURSE\PYTHON\open cv how to work with\statue.jpg', cv.IMREAD_GRAYSCALE)
# clahe = cv.createCLAHE(clipLimit= 2.0 , tileGridSize=(8,8))
# cl1 = clahe.apply(img)
# cv.imshow('test',cl1)
# cv.waitKey(0)

#-----------------------------------------------------------------------------------
#نحوه ی پیدا کردن یک ابجکت در یک عکس با استفاده از الگوریتم 
# template matching 

# img_rgb = cv.imread(r'D:\9-PYTHON COURSE\PYTHON\open cv how to work with\mario.jpg')
# img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
# template  = cv.imread(r'D:\9-PYTHON COURSE\PYTHON\open cv how to work with\coin.jpg', cv.IMREAD_GRAYSCALE)

# w, h = template.shape[::-1]
 
# result  = cv.matchTemplate(img_gray,template,cv.TM_CCOEFF_NORMED)
# threshold = 0.7
# loc = np.where( result  >= threshold)
# for pt in zip(*loc[::-1]):
#     cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
 
# cv.imshow('test', img_rgb)
# cv.waitKey(0)
# cv.destroyAllWindows()

#-----------------------------------------------------------------------------------
# Hough Line Transform
# با استفاده از این میتونیم خطوط را پیدا کنیم 

# import cv2 as cv 
# import numpy as np 

# img = cv.imread(cv.samples.findFile(r'D:\9-PYTHON COURSE\PYTHON\open cv how to work with\box.jpg'))
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# edges = cv.Canny(gray, 50, 150, apertureSize = 3)

# lines = cv.HoughLines(edges, 2, np.pi/200, 200)

# for line in lines :
#     rho, theta = line[0]
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a * rho
#     y0 = b * rho
#     x1 = int(x0 + 1000 * (-b))
#     y1 = int(y0 + 1000 * (a))
#     x2 = int(x0 - 1000 * (-b))
#     y2 = int(y0 - 1000 * (a))
    
#     cv.line(img, (x1,y1), (x2,y2), (0,0,255), 2)
    
# cv.imshow('test', img)
# cv.waitKey(0)
# cv.destroyAllWindows()

# روش دوم 

# img = cv.imread(cv.samples.findFile(r'D:\9-PYTHON COURSE\PYTHON\open cv how to work with\box.jpg'))
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# edges = cv.Canny(gray, 50, 150, apertureSize = 3)
# lines = cv.HoughLinesP(edges, 1, np.pi/180, 40, minLineLength = 50, maxLineGap = 10)

# for line in lines: 
#     x1, y1, x2, y2 = line[0]
#     cv.line(img, (x1,y1), (x2,y2), (0,255,0), 2)
    
# cv.imshow('test', img)
# cv.waitKey(0)
# cv.destroyAllWindows()
#-----------------------------------------------------------------------------------
# برای شناسایی دایره و کشیدن خط دور آن 

# img = cv.imread(r'D:\9-PYTHON COURSE\PYTHON\open cv how to work with\test.jpg', cv.IMREAD_GRAYSCALE)
# img = cv.medianBlur(img, 5)
# cimg = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
# circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 20, param1 = 50, param2 = 30, minRadius = 0, maxRadius = 0)

# circles = np.uint16(np.around(circles))

# for i in circles[0,:]:
#     cv.circle(cimg, (i[0], i[1]), i[2], (0,255,0),2)
#     cv.circle(cimg,(i[0], i[1]),2,(0,0,255),3)
    
# cv.imshow('test', cimg)
# cv.waitKey(0)
# cv.destroyAllWindows()
#-----------------------------------------------------------------------------------
#پیدا کردن پستی و بلندی ها و نقاط چسبیده بهم
# import numpy as np 
# import cv2 as cv
# from matplotlib import pyplot as plt 
# اول دایره های بهم چسبیده رو پیدا میکنیم 
# قسمت های اضافی رو پاک میکنیم 
# نقاط متصل بهم رو پیدا میکنیم 
# سکه رو از پس زمینه جدا میکنیم 

# img = cv.imread(r'D:\9-PYTHON COURSE\PYTHON\open cv how to work with\coins.jpg')
# assert img is not None, "file could not be read, check with os.path.exists()"
# gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
# kernel = np.ones((3,3),np.uint8)
# opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)
# sure_bg = cv.dilate(opening,kernel,iterations=3)
# dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
# ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)
# sure_fg = np.uint8(sure_fg)
# unknown = cv.subtract(sure_bg,sure_fg)
# ret, markers = cv.connectedComponents(sure_fg)
# markers = markers+1
# markers[unknown==255] = 0
# markers = cv.watershed(img,markers)
# img[markers == -1] = [255,0,0]
# markers_8u = cv.convertScaleAbs(markers)
# cv.imshow('Watershed Segmentation', img)
# cv.imshow('Markers', markers_8u)

# cv.waitKey(0)
# cv.destroyAllWindows()

#-----------------------------------------------------------------------------------
#جدا کردن یک ابجکت از درون یک عکس 

# import numpy as np 
# import cv2 as cv
# from matplotlib import pyplot as plt 

# img = cv.imread(r'D:\9-PYTHON COURSE\PYTHON\open cv how to work with\messi.jpg')
# mask = np.zeros(img.shape[:2], np.uint8)
# bgdmodel = np.zeros((1,65), np.float64)
# fgdmodel = np.zeros((1,65), np.float64)
# rect = (50, 50, 450, 290)
# cv.grabCut(img, mask, rect, bgdmodel, fgdmodel, 5, cv.GC_INIT_WITH_RECT)
# mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
# img = img * mask2[:,:,np.newaxis]
# plt.imshow(img),plt.colorbar(), plt.show()

#-----------------------------------------------------------------------------------
# feature detection 
"""
تو این فصل راجب یادگیری تصاویر،شناسایی گوشه ها و شناسایی تصاویر صحبت میکنیم 
"""
# import numpy as np 
# import cv2 as cv 

# filename = r'D:\9-PYTHON COURSE\PYTHON\open cv how to work with\chess.jpg'
# img = cv.imread(filename)

# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# gray = np.float32(gray)

# dst = cv.cornerHarris(gray, 2, 3, 0.04)
# dsy = cv.dilate(dst, None)
# img[dst > 0.01 * dst.max()] = [0,0,255]

# cv.imshow('test', img)
# cv.waitKey(0)
# cv.destroyAllWindows()
#-----------------------------------------------------------------------------------
# پیدا کردن نقطه روی عکس 
#پیدا کردن keypoint
# import numpy as np
# import cv2 

# img = cv2.imread('original.jpg')


# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# sift = cv2.SIFT_create()
# kp = sift.detect(gray, None)
# img = cv2.drawKeypoints(gray, kp, img)
# cv2.imshow('sift_keypoints.jpg', img)
# cv2.waitKey(0)
#-----------------------------------------------------------------------------------
#برای پیدا کردن تشابهات در عکس مثل دو نقطه سفید یا سیاه در عکس 
 
# import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt

# img = cv.imread('fly.jpg', cv.IMREAD_GRAYSCALE)

# img2 = cv.drawKeypoints(img, kp, None,(255, 0, 0), 4)
# plt.imshow(img2), plt.show()
#-----------------------------------------------------------------------------------
# fast algorithm
# سریعترین راه برای پیدا کردن گوشه ها 

# import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt 

# img = cv.imread(r'angels.jpg', cv.IMREAD_GRAYSCALE)
# fast = cv.FastFeatureDetector()
# kp = fast.detect(img, None)
# img2 = cv.drawKeypoints(img, kp, None, color=(255,0,0))
# print( "Threshold: {}".format(fast.getThreshold()) )
# print( "nonmaxSuppression:{}".format(fast.getNonmaxSuppression()) )
# print( "neighborhood: {}".format(fast.getType()) )
# print( "Total Keypoints with nonmaxSuppression: {}".format(len(kp)) )
# cv.imshow('test', img2)
# fast.setNonmaxSuppression(0)
# kp = fast.detect(img, None)
# print("total keypoints without nonmax suppression :{}".format(len(kp)))
# img3 = cv.drawKeypoints(img, kp, None, color=(255,0,0))
# cv.imshow('test2', img3)
#-----------------------------------------------------------------------------------
# 2 images detector

# import numpy as np
# import cv2 as cv
# import matplotlib.pyplot as plt
 
# img1 = cv.imread('pic1.jpg',cv.IMREAD_GRAYSCALE) # queryImage
# img2 = cv.imread('pic2.jpg',cv.IMREAD_GRAYSCALE) # trainImage
 
# # Initiate SIFT detector
# sift = cv.SIFT.create()
 
# # find the keypoints and descriptors with SIFT
# kp1, des1 = sift.detectAndCompute(img1,None)
# kp2, des2 = sift.detectAndCompute(img2,None)
 
# # BFMatcher with default params
# bf = cv.BFMatcher()
# matches = bf.knnMatch(des1,des2,k=2)
 
# # Apply ratio test
# good = []
# for m,n in matches:
#  if m.distance < 0.75*n.distance:
#     good.append([m])
 
# # cv.drawMatchesKnn expects list of lists as matches.
# img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
 
# plt.imshow(img3),plt.show()


# second way 
# import numpy as np 
# import cv2 as cv 
# import matplotlib.pyplot as plt 

# img1 = cv.imread(r'pic1.jpg', cv.IMREAD_GRAYSCALE)
# img2 = cv.imread(r'pic2.jpg', cv.IMREAD_GRAYSCALE)

# sift = cv.SIFT.create()

# kp1, des1 = sift.detectAndCompute(img1, None)
# kp2, des2 = sift.detectAndCompute(img2, None)

# bf = cv.BFMatcher()
# matches = bf.knnMatch(des1, des2, k=2)

# good = []

# for m, n in matches:
#     if m.distance < 0.75 * n.distance:
#         good.append([m])
        
# img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# plt.imshow(img3)
# plt.show()


# third way 

# import numpy as np 
# import cv2 as cv
# import matplotlib.pyplot as plt 

# img1 = cv.imread(r'pic1.jpg', cv.IMREAD_GRAYSCALE)
# img2 = cv.imread(r'pic2.jpg', cv.IMREAD_GRAYSCALE)

# sift = cv.SIFT.create()

# kp1, des1 = sift.detectAndCompute(img1, None)
# kp2, des2 = sift.detectAndCompute(img2, None)


# FLANN_INDEX_KDTREE = 1
# index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
# search_params = dict(checks=50) # or pass empty dictionary
 
# flann = cv.FlannBasedMatcher(index_params,search_params)
 
# matches = flann.knnMatch(des1,des2,k=2)
 
# matchesMask = [[0,0] for i in range(len(matches))]
 
# for i,(m,n) in enumerate(matches):
#     if m.distance < 0.7*n.distance:
#         matchesMask[i]=[1,0]
 
# draw_params = dict(matchColor = (0,255,0), singlePointColor = (255,0,0), matchesMask = matchesMask, flags = cv.DrawMatchesFlags_DEFAULT)
# img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
 
# plt.imshow(img3,),plt.show()
#-----------------------------------------------------------------------------------
# import numpy as np
# import cv2 as cv 
# import matplotlib.pyplot as plt

# MIN_MATCH_COUNT = 10

# img1 = cv.imread(r'pic1.jpg', cv.IMREAD_GRAYSCALE)
# img2 = cv.imread(r'pic2.jpg', cv.IMREAD_GRAYSCALE)

# sift = cv.SIFT.create()

# kp1, des1 = sift.detectAndCompute(img1, None)
# kp2, des2 = sift.detectAndCompute(img2, None)

# FLANN_INDEX_KDTREE = 1 
# index_param = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
# search_params = dict(checks = 50)
# flann = cv.FlannBasedMatcher(index_param, search_params)
# matches = flann.knnMatch(des1, des2, k=2)
# good = []
# for m, n in matches:
#     if m.distance < 0.7 * n.distance:
#         good.append(m)

# if len(good)>MIN_MATCH_COUNT:
#  src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
#  dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
 
#  M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
#  matchesMask = mask.ravel().tolist()
 
#  h,w = img1.shape
#  pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
#  dst = cv.perspectiveTransform(pts,M)
 
#  img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
 
# else:
#  print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
#  matchesMask = None

# draw_params = dict(matchColor = (0,255,0), # draw matches in green color
#  singlePointColor = None,
#  matchesMask = matchesMask, # draw only inliers
#  flags = 2)
 
# img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
 
# plt.imshow(img3, 'gray'),plt.show()
#-----------------------------------------------------------------------------------
#تنظیم کردن رنگ  و نور عکس
# from __future__ import print_function
# from __future__ import division
# import cv2 as cv
# import numpy as np 
# import argparse
# import os 

# def loadExposureSeq(path):
#     images = []
#     times = []
#     with open(os.path.join(path, 'list.txt')) as f:
#         content = f.readlines()
#         for line in content:
#             tokens = line.split()
#             images.append(cv.imread(os.path.join(path, tokens[0])))
#             times.append(1 / float(tokens[1]))
            
#             return images, np.asarray(times, dtype=np.float32)
#         parser = argparse.ArgumentParser(description='code for high dynamic range imaging tutorial.')
#         parser.add_argument('__input', type=str, help='path to the directory that contains images and exposure times.')
#         args = parser.parse_args()
        
#         if not args.input:
#             parser.print_help()
#             exit(0)
            
#         images, times = loadExposureSeq(args.input)
#         calibrate = cv.createCalibrateDebevec()
#         response = calibrate.process(images, times)
#         merge_debevec = cv.createMergeDebevec()
#         hdr = merge_debevec.process(images, times, response)
#         tonemap = cv.createTonemap(2.2)
#         ldr = tonemap.process(hdr)
#         merge_mertens = cv.createMergeMertens()
#         fusion = merge_mertens.process(images)
        
#     cv.imwrite('fusion.png', fusion * 255)
#     cv.imwrite('ldr.png', ldr * 255)
#     cv.imwrite('hdr.hdr', hdr)
#-----------------------------------------------------------------------------------
# background subtraction 

# from __future__ import print_function
# import cv2 as cv
# import argparse
 
# parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
#  OpenCV. You can process both videos and images.')
# parser.add_argument('--input', type=str, help='Path to a video or a sequence of ``image.', default=r'D:\9-PYTHON COURSE\PYTHON\open cv how to work with\subtraction.jpg')
# parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
# args = parser.parse_args()
 
 
# if args.algo == 'MOG2':
#  backSub = cv.createBackgroundSubtractorMOG2()
# else:
#  backSub = cv.createBackgroundSubtractorKNN()
 
 
 
# capture = cv.VideoCapture(cv.samples.findFileOrKeep(args.input))
# if not capture.isOpened():
#  print('Unable to open: ' + args.input)
#  exit(0)
 
 
# while True:
#     ret, frame = capture.read()
#     if frame is None:
#      break
 
 
 
#     fgMask = backSub.apply(frame)
 
 
 
#     cv.rectangle(frame, (10, 2), (200,40), (255,255,255),-1)
#     cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
#     cv.imshow('Frame', frame)
#     cv.imshow('FG Mask', fgMask)
#     cv.waitKey()
#-----------------------------------------------------------------------------------
#پیدا کردن نقاطی در عکس که تراکم بیشتری نسبت به بقیه جاها داره
# import numpy as np 
# import cv2 as cv 
# import argparse

# parser = argparse.ArgumentParser(description='this sample demonstrates th ecamshift algorith.')
# parser.add_argument('image', type= str,help=r'D:\9-PYTHON COURSE\PYTHON\open cv how to work with\car.jpg')
# args = parser.parse_args()

# cap = cv.VideoCapture(args.image)
# ret,frame = cap.read()
# x, y, w, h = 300, 200, 100, 50 
# track_window = (x, y, w, h)
# roi = frame[y:y+h, x:x+w]
# hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
# mask = cv.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
# roi_hist = cv.calcHist([hsv_roi],[0],mask,[180],[0,180])
# cv.normalize(roi_hist,roi_hist,0,255,cv.NORM_MINMAX)
# term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1 )
# while(1):
#     ret, frame = cap.read()
    
#     if ret == True:
#         hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
#         dst = cv.calcBackProject([hsv],[0],roi_hist,[0,180],1)
#         ret, track_window = cv.CamShift(dst, track_window, term_crit)
#         pts = cv.boxPoints(ret)
#         pts = np.int0(pts)
#         img2 = cv.polylines(frame,[pts],True, 255,2)
#         cv.imshow('img2',img2)
 
#         k = cv.waitKey(30) & 0xff
#     if k == 27:
#         break
#     else:
#         break
#-----------------------------------------------------------------------------------
#نشان دادن مسیر حرکت یک ابجکت در تصویر از فریم اول به فریم دوم 
# import numpy as np 
# import cv2 as cv
# import argparse

# parser =argparse.ArgumentParser(description='Nothings important here.')
# parser.add_argument('test', type= str, help=r'car.jpg')
# args = parser.parse_args()


# cap = cv.VideoCapture(args.image)
# feature_params = dict( maxCorners = 100, qulitylevel = 0.3, mindistance= 7, blocksize = 7 )
# lk_params = dict(winsize = (15, 15), maxlevel  = 2, criteria = (cv.TERM_CRITERIA_EPS | cv.TermCriteria_COUNT, 10, 0.03))
# color = np.random.randint(0, 255, (100, 3))
# ret, old_frame = cap.read()
# old_gray =  cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
# p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
# mask = np.zeros_like(old_frame)
# while(1):
#     ret,frame = cap.read()
#     if not ret:
#         print('no frames grabbed')
#         break
#     frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#     p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
#     if p1 is not None:
#         good_new = p1[st==1]
#         good_old = p0[st==1]
        
        
#         for i, (new, old) in enumerate(zip(good_new, good_old)):
#             a,b = new.ravel()
#             c, d = old.ravel()
#             mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
#             frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), 2)
#             img = cv.add(frame, mask)
#             cv.imshow('frame', img)
#             cv.waitKey(0)
#-------------------------------------------------------------------------------------------
#شناسایی گردی چهره و چشم ها 
#cascadeclassifier load => to load a xml classifier file. it can be either a haar or a lbp classifier 
# cascade classifier- detectmultiscale => to perform the detection 
# cascade clssifier => class to detect objects in a video stream.

#برای آمورش ماشین باید تعداد زیادی عکس به ماشین بدیم عکس های مثبت، عکس های منفی،عکس های فاقد صورت تا بتونه تشخیص بده پس از آن باید موارد لازم رو از عکس بکشیم بیرون 
#برای شناسایی موارد مورد نیاز میایم تعداد پیکسلای زیر مستطیل رو بدست میاریم 

# from __future__ import print_function
# import cv2 as cv
# import argparse

# def detectanddisplay(frame):
#     frame_gray = cv.cvtColor(frame, cv.COLOR_BayerBG2GRAY)
#     frame_gray = cv.equalizeHist(frame_gray)
    
#     faces = face_cascade.detectMultiScale(frame_gray)
#     for(x, y, w, h) in faces:
#         center = (x + w//2, y + h//2)
#         frame = cv.ellipse(frame, center,(w // 2 , h // 2), 0, 0 , 360, (255, 0 , 255), 4)
#         faceROI = frame_gray[y:y+h, x:x+w]
#         eyes = eyes_cascade.detectMultiScale(faceROI)
#         for (x2, y2, w2, h2) in eyes:
#             eye_center = (x + x2 + w2//2 , y + y2 + h2 //2)
#             radius = int(round((w2 +h2 ) * 0.25))
#             frame = cv.circle(frame, eye_center, radius, (255, 0, 0), 4)
#             frame  = cv.circle(frame, eye_center, radius, (255, 0, 0) , 4)
#         cv.imshow('Capture  - Face detectioon ', frame)
# parser = argparse.ArgumentParser(description = 'Code for Cascade Classifier tutorial.')
# parser.add_argument('--face_cascade', default = r'D:\9-PYTHON COURSE\PYTHON\open cv how to work with\face.jpg')
# parser.add_argument('--eyes_cascade', default=  r'D:\9-PYTHON COURSE\PYTHON\open cv how to work with\face.jpg')
# parser.add_argument('-- camera', help='Camera divide number.' , type= int, default=0)
# args = parser.parse_args()

# face_cascade_name = args.face_cascade
# eyes_cascade_name = args.eyes_cascade

# face_cascade = cv.CascadeClassifier()
# eyes_cascade = cv.CascadeClassifier()

# if not face_cascade.load(cv.samples.findFile(face_cascade_name)):
#     print('--(!) Error loading face cascade')
#     exit(0)
# if not eyes_cascade.load(cv.samples.findFile(eyes_cascade_name)):
#     print('--(!) Error loading eyes cascade')
#     exit(0)

# camera_device = args.camera
# cap = cv.VideoCapture(camera_device)
# if not cap.isOpened:
#     print('--(!) Error opening video capture')
#     exit(0)
    
# while True:
#     ret, frame = cap.read()
#     if frame is None:
#         print('--(!) No captured frame -- Break!')
#         break
    
#     detectanddisplay(frame)
    
#     if cv.waitKey(10) == 27 :
#         break

#----------------------------------------------------------------------------------------------------- 
# camera calibration and 3d reconstruction

# import numpy as np
# import cv2 as cv
# import glob

# rows = 6
# cols = 7

# criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# objp = np.zeros((rows*cols, 3), np.float32)
# objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
# objpoints = []
# imgpoints = []
# images_path = r'D:\9-PYTHON COURSE\PYTHON\open cv how to work with\images'
# images = glob.glob(images_path + '\*.png')

# for fname in images:
#     img = cv.imread(fname)
    
#     if img is None:
#         print(f'Failed to load the image : {fname}')
#         continue
    
#     gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#     ret, corners = cv.findChessboardCorners(gray, (cols, rows), None)
    
#     if ret:
#         corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
#         objpoints.append(objp)
#         imgpoints.append(corners2)
#         cv.drawChessboardCorners(img, (cols, rows), corners2, ret)
#         cv.imshow('img', img)
#         cv.waitKey(0)
#     else:
#         print(f'Failed to find chessboard corners in image : {fname}')

# cv.destroyAllWindows()

# # Get the shape of the last image
# image_shape = gray.shape[::-1]

# ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, image_shape, None, None)
# print('Camera Matrix:')
# print(mtx)
# print('Distortion Coefficients:')
# print(dist)
# mean_error = 0
# for i in range(len(objpoints)):
#     imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
#     error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/ len(imgpoints2)
#     mean_error += error
# print('Total error : {}'.format(mean_error / len(objpoints)))
#-----------------------------------------------------------------------------------------------------
# import numpy as np
# import cv2 as cv
# import glob

# with np.load('B.npz') as x :
#     mtx, dist, _, _ = [x[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]
    
# def draw(img, corners, imgpts):
#     imgpts = np.int32(imgpts).reshape(-1,2)

#     # draw ground floor in green
#     img = cv.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)
 
#     # draw pillars in blue color
#     for i,j in zip(range(4),range(4,8)):
#         img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
 
#     # draw top layer in red color
#     img = cv.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)
 
#     return img

# criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# objp = np.zeros((6 * 7, 3), np.float32)
# objp[:,:2] = np.mgrid[0:7, 0:6].T.reshape(-1,2)
# axis = np.float32([[0, 0, 0], [0, 3, 0], [3, 3, 0], [3, 0, 0], [0, 0, -3], [0, 3, -3], [3, 3, -3], [3, 0, -3]])
# for fname in glob.glob(r'chess.png'):
#     img =  cv.imread(fname)
#     gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#     ret, corners = cv.findChessboardCorners(gray, (7, 6), None)
    
#     if ret == True:
#         corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
#         ret, rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)
#         imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
#         img = draw(img, corners2, imgpts)
#         cv.imshow('img', img)
#         k = cv.waitKey(0)
#         if k == ord('s'):
#             cv.imshow('test', fname[:6] + '.png', img)
# cv.destroyAllWindows()

#-----------------------------------------------------------------------------------------------------
# تهیه نقشه نابرابری از عکس 
# import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt 

# imgl = cv.imread(r'D:\9-PYTHON COURSE\PYTHON\open cv how to work with\statue.jpg', cv.IMREAD_GRAYSCALE)
# imgr = cv.imread(r'D:\9-PYTHON COURSE\PYTHON\open cv how to work with\statue.jpg', cv.IMREAD_GRAYSCALE)

# stereo = cv.StereoBM.create(numDisparities = 16, blockSize = 15)
# disparity = stereo.compute(imgl, imgr)
# plt.imshow(disparity, 'gray')
# plt.show()
#-----------------------------------------------------------------------------------------------------
# Machine Learning
 
# شناسایی نزدیک ترین اشیا
# با استفاده از الگوریتم KNN
# k nearest neighbour
# نزدیک ترین و شبیه ترین طبقه بندی عکس
#روش وزن دهی و تشخیص به صورت زیر میباشد
"""
روش وزن دهی به این صورت هست که فکر کن 10 تا خونه تو دو محل متفاوت قرار دارند حالا یه همسایه جدید میاد و میخواد
خونه بسازه پس منطقا هر جا که نزدیک تر بود به هر کدوم از این دو محل جزو اون محل قرار میگیره
در knn هم به این صورت هست که وزن دهی میکنیم اما وزن دهی به چه صورت هست:
میایم فاطله ابجکت ها رو به اطراف محاسبه میکنیم و در نهایت ابجکت جدیدمون به هرکدوم از اون ابجکت ها نزدیک تر بود
اون وزن بیشتری میگیره و برعکس ان
"""

# import cv2 as cv
# import numpy as np 
# import matplotlib.pyplot as plt 

# # بیست و پنج تا پک دوتایی که عدد داخلش رندوم بین صفر تا صد هست با فرمت فلوت 32 بده 
# trainData = np.random.randint(0,100,(25, 2)).astype(np.float32)
# responses = np.random.randint(0,2,(25, 1)).astype(np.float32)

# # red scope
# red = trainData[responses.ravel()==0]
# plt.scatter(red[:, 0], red[:, 1], 10, 'r', '^')

# # blue scope 
# blue = trainData[responses.ravel()==1]
# plt.scatter(blue[:, 0], blue[:, 1], 10, 'b', )

# # show the location of each blue, red point in the pic
# # for i, point in enumerate(red):
# #     print(f'Red poinr {i +1} : ({point[0]}, {point[1]})')
# # for i, point in enumerate(blue):
# #     print(f'blue point {i+1} : ({point[0]}, {point[1]})')

# newComer = np.random.randint(0, 100, (1, 2)).astype(np.float32)
# knn = cv.ml.KNearest.create()
# knn.train(trainData, cv.ml.ROW_SAMPLE, responses)
# ret, results, neighbours, dist = knn.findNearest(newComer, 3)

# print('result: {}\n'.format(results))
# print('neighbours: {}\n'.format(neighbours))
# print('distance: {} \n'.format(dist))

# plt.show()

#-----------------------------------------------------------------------------------------------------
#ساخت برنامه ای که بتواند دست خط را شناسایی کند 
"""
برای اینک ار به دو نوع داده نیاز داریم، 1- داده تمرینی و 2- داده تست 
"""
# import numpy as np 
# import cv2 as cv

# img = cv.imread('digits.py')
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cells = [np.hsplit(row,100) for row in np.vsplit(gray, 50)]
# x = np.array(cells)

# train = x[:, :50].reshape(-1, 400).astype(np.float32)
# test = x[:, 50:100].reshape(-1, 400).astype(np.float32)

# k = np.arange(10)
# train_labels = np.repeat(k, 250)[:, np.newaxis]
# test_labels = train_labels.copy()

# knn = cv.ml.KNearest.create()
# knn.train(train, cv.ml.ROW_SAMPLE, train_labels)
# ret, result, neighbours, dist = knn.findNearest(test, k=5)

# matches = result == test_labels
# correct = np.count_nonzero(matches)
# accuracy = correct * 100.0 / result.size

# np.savez('knn_data.npz', train - train, train_labels = train_labels)
# with np.load('knn_data.npz') as data:
#     print(data.files)
#     train = data['train']
#     train_labels = data['train_labels']
#-----------------------------------------------------------------------------------------------------
#شناسایی و تشخیص دست خط با استفاده ازز ocr/svm
import cv2 as cv
import numpy as np

sz = 20
bin_n = 16

affine_flages = cv.WARP_INVERSE_MAP | cv.INTER_LINEAR

def deskew(img):
    m = cv.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * sz * skew ]])
    img = cv.warpAffine(img, M, (sz, sz), flags=affine_flages)
    return img 

def hog(img):
    gx = cv.Sobel(img, cv.CV_32F, 1, 0)
    gy = cv.Sobel(img, cv.CV_32F, 0, 1)
    mag, ang = cv.cartToPolar(gx, gy)
    bins = np.int32(bin_n * ang / (2 * np.pi))
    bin_cells = bins[:10, :10], bins[10:, :10] , bins[:10, 10:], bins[10:, 10:]
    mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)
    return hist

img = cv.imread(cv.samples.findFile('digits.png'), 0)
if img is None:
    raise Exception('we need th digits.png image from samples/ data here! ')

cells = [np.hsplit(row, 100) for row in np.vsplit(img, 50)]

train_cells = [ i[:50] for i in cells ]
test_cells = [ i[50:] for i in cells]
 
 
 
deskewed = [list(map(deskew,row)) for row in train_cells]
hogdata = [list(map(hog,row)) for row in deskewed]
trainData = np.float32(hogdata).reshape(-1,64)
responses = np.repeat(np.arange(10),250)[:,np.newaxis]
 
svm = cv.ml.SVM_create()
svm.setKernel(cv.ml.SVM_LINEAR)
svm.setType(cv.ml.SVM_C_SVC)
svm.setC(2.67)
svm.setGamma(5.383)
 
svm.train(trainData, cv.ml.ROW_SAMPLE, responses)
svm.save('svm_data.dat')
 
 
 
deskewed = [list(map(deskew,row)) for row in test_cells]
hogdata = [list(map(hog,row)) for row in deskewed]
testData = np.float32(hogdata).reshape(-1,bin_n*4)
result = svm.predict(testData)[1]
 
 
mask = result==responses
correct = np.count_nonzero(mask)
print(correct*100.0/result.size)
