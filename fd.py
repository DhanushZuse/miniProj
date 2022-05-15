import cv2

trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cam = cv2.VideoCapture(0)

while True:

    successful, img = cam.read()

    #img = cv2.imread('sam.jpg')

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_loc = trained_face_data.detectMultiScale(gray_img)

    for (x,y, w, h) in face_loc:

        cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,255), 2)



    print(face_loc)

    cv2.imshow('cam',img)

    cv2.waitKey(1)