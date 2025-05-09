'''
کاری که این الکوریتم انجام میده اینه که میاد در حالت عادی دو تا دیتای موجود رو به وسیله ی خط هایی از هم جدا میکنه  و مرزی بین این دیتا با اون دیتا پیدا میکنه
و بعد از بین خط هایی که پیدا شده میاد اونی که از هر دو طرف به دیتا ها دورتر هست رو انتخاب میکنه چون ممکنه هرکدوم از داده ها نویز هایی داشته باشند که خط ها رو تحت تاثیر قرار بده
پس باید دورترین انتخاب بشه.

در حالتی که از این الگوریتم استفاده نکنیم باید تک تک داده ها را پیدا کنیم و فاصله از طرفین رو حساب کنیم 

'''
#--------------------------------------------------------------------------------------------
#OCR
'''
در knn 
ما مستقیما تجمع پیکسل ها رو حساب میکنیم اما با استفاده از OCR
اینجا عکس اصلی رو میدیم و عکس شناسایی شده رو میده
بعد از این مرحله نیاز داریم تا توصیفگر هر سلول رو بدست بیاریم
برای این کار  مشتقات هر سلول را در مختصات ایکس و ایگرگ بدست میاریم سپس قدر و جهت هر یک پیکسل را بدست میاریم
این کار هر مقدار را به 16 مقدار عددی تبدیل میکنه و عکس را به 4 مربع زیرین تبدیل میکنه در نهایت
شروع میکنیم به تبدیل دیتاست خودمون به سلول های تکی  برای هر شماره 250 سلول هستند که ترین میشن  و
 250 دیتا هستند برای تست 
'''

import cv2 as cv
import numpy as np

sz = 20 
bin_n = 16
affine_flags = cv.WARP_INVERSE_MAP | cv.INTER_LINEAR

def deskew(img):
    m = cv.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv.warpAffine(img,M,(SZ, SZ),flags=affine_flags)
    return img

 
def deskew(img):
    m = cv.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv.warpAffine(img,M,(SZ, SZ),flags=affine_flags)
    return img
 
 
 
def hog(img):
    gx = cv.Sobel(img, cv.CV_32F, 1, 0)
    gy = cv.Sobel(img, cv.CV_32F, 0, 1)
    mag, ang = cv.cartToPolar(gx, gy)
    bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)     # hist is a 64 bit vector
    return hist
 
 
img = cv.imread(cv.samples.findFile(r'D:\9-PYTHON COURSE\PYTHON\open cv how to work with\ocr.jpg'),0)
if img is None:
    raise Exception("we need the digits.png image from samples/data here !")
 
 
cells = [np.hsplit(row,100) for row in np.vsplit(img,50)]
 
# First half is trainData, remaining is testData
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