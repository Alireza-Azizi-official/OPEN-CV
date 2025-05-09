'''
در این الگوریتم تعداد زیادی دیتا داریم برای راحت تر شدن کار میایم دیتا ها رو دسته بندی میکنیم 
این دسته بندی به ما کمک میکنه داده های مرتب تر و راحت تری داشته باشیم
منتهی همیشه داده هایی هستند که جدا افتادن اولین کار اینه که میایم دو داده رو رندوم انتخاب میکنیم و فاصله انها را حساب میکنیم 
و زمانی که حساب کردیم نزدیک ترین گروه به اون مقیاس رو به اون داده ارتباط میدیم و دسته بندی میکنیم 

'''
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
 
X = np.random.randint(25,50,(25,2))
Y = np.random.randint(60,85,(25,2))
Z = np.vstack((X,Y))
 
# convert to np.float32
Z = np.float32(Z)
 
# define criteria and apply kmeans()
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret,label,center=cv.kmeans(Z,2,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)
 
# Now separate the data, Note the flatten()
A = Z[label.ravel()==0]
B = Z[label.ravel()==1]
 
# Plot the data
plt.scatter(A[:,0],A[:,1])
plt.scatter(B[:,0],B[:,1],c = 'r')
plt.scatter(center[:,0],center[:,1],s = 80,c = 'y', marker = 's')
plt.xlabel('Height'),plt.ylabel('Weight')
plt.show()