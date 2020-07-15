import cv2
img = cv2.imread('C:\\Users\\venog\\Desktop\\Diploma\\facenet\\data\\images\\train_raw\\Eva\\Eva_0000.JPG') # load a dummy image
while(1):
    cv2.imshow('img',img)
    k = cv2.waitKey(33)
    if k==27:    # Esc key to stop
        break
    elif k==-1:  # normally -1 returned,so don't print it
        continue
    else:
        print(k) # else print its value