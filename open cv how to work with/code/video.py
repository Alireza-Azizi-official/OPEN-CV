# # how to work with video 
# # from camera 

# import numpy as np
# import cv2 as cv 

# cap = cv.VideoCapture(0)

# if not cap.isOpened():
#     print('no camera detected')
#     exit()
    
# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()
    
#     # if frame is read correctly ret is True
#     if not ret:
#         print('cant recieve frame (stream end?). exiting.....')
        
#     # our operations on the frame come here
#     gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#     cv.imshow('test name', gray)
    
#     # Display the resulting frame
#     if cv.waitKey(1) == ord('q'):
#         break
    

# # when everything's done, release the capture
# cap.release()
# cv.destroyAllWindows()

# #-------------------------------------------------------------------------

# # from video

# import numpy as np
# import cv2 as cv 

# cap = cv.VideoCapture(r'D:\9-PYTHON COURSE\PYTHON\open cv how to work with\test video.mp4')

# while cap.isOpened():
#     ret, frame = cap.read()
    
#     if not ret:
#         print("can't recieve frame (stream end?). Exiting.....")
#         break
    
#     gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#     cv.imshow('test name', gray)
    
#     if cv.waitKey(1) == ord('q'):
#         break

# cap.release()
# cv.destroyAllWindows()


# #----------------------------------------------------------------------------
# # to save the video

# # باید یک آبجکت ویدیو رایتر بنویسیم

# import numpy as np
# import cv2 as cv

# cap = cv.VideoCapture(r'D:\9-PYTHON COURSE\PYTHON\open cv how to work with\test video.mp4')

# # define the codec and create videowriter object
# fourcc = cv.VideoWriter.fourcc(*'XVID')
# out = cv.VideoWriter('out.mp4', fourcc, 20.0, (640, 480))

# while cap.isOpened():
#     ret, frame = cap.read()
    
#     if not ret:
#         print('Cant receive frame (stream end?). exiting....')
#         break
    
#     frame = cv.flip(frame, 0)
#     out.write(frame)
    
#     cv.imshow('frame', frame)
#     if cv.waitKey(1)== ord('q'):
#         break
    
# cap.release()
# out.release()
# cv.destroyAllWindows()
# #----------------------------------------------------------------------------
##تشخیص رنگ و تبدیل رنگ ها و گرفتن ابجکت مشخص 
##تبدیل hsv 
## برای بدست اوردن hsv کد زیر رو میزنیم 

# import numpy as np
# import cv2 as cv


# # green = np.uint8([[[0,255,0]]])
# # hsv_green = cv.cvtColor(green, cv.COLOR_BGR2HSV)
# # print(hsv_green)

# cap = cv.VideoCapture(0)

# while(1):
#     _ , frame = cap.read()
#     hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    
#     lower_blue = np.array([100,50,50])
#     upper_blue = np.array([140, 255, 255])
    
#     mask = cv.inRange(hsv, lower_blue, upper_blue)
#     res = cv.bitwise_and(frame, frame, mask=mask)
    
#     cv.imshow('frame', frame)
#     cv.imshow('mask',mask)
#     cv.imshow('res',res)
#     k = cv.waitKey(5)
#     if k == ord('q'):
#         break

# cv.destroyAllWindows()

# #----------------------------------------------------------------------------
