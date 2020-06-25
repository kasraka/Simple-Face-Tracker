import cv2
import dlib

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cv2.namedWindow('Video', cv2.WINDOW_AUTOSIZE)
# To use a video, insert the file's name accordingly like this "capture = cv2.VideoCapture('File.mp4')"
frameCounter, faceNumber, img_counter = 0, 0, 0
faceTrack, face2Delete = {}, []
while True:
    _, frame = capture.read()
    frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    k = cv2.waitKey(33)
    for iD in faceTrack.keys():
        quality = faceTrack[iD].update(frame)
        if quality<8:
            face2Delete.append(iD)
    for iD in face2Delete:
        faceTrack.pop(iD, None)
    if frameCounter % 5 == 0:
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(frame, 1.2, 5)
        for x, y, w, h in faces:
            x_center = x + w/2
            y_center = y +h/2
            theID = None
            for iD in faceTrack.keys():
                trackedPosition = faceTrack[iD].get_position()
                tracker_x = trackedPosition.left()
                tracker_y = trackedPosition.top()
                tracker_w = trackedPosition.width()
                tracker_h = trackedPosition.height()
                tracker_x_center = tracker_x + tracker_h/2
                tracker_y_center = tracker_y + tracker_w/2
                if (x <= tracker_x_center <= (x + w)) and (y <= tracker_y_center <= (y + h)) and (tracker_x <= x_center <= (tracker_x + tracker_w)) and (tracker_y <= y_center <= (tracker_y + tracker_h)):
                    theID = iD
            if theID is None:
                tracker = dlib.correlation_tracker()
                tracker.start_track(frame2, dlib.rectangle(x-20, y-20, x+w+20, y+h+20))
                faceTrack[faceNumber] = tracker
                faceNumber +=1
    for iD in faceTrack.keys():
        trackedPosition = faceTrack[iD].get_position()
        tracker_x = int(trackedPosition.left())
        tracker_y = int(trackedPosition.top())
        tracker_w = int(trackedPosition.width())
        tracker_h = int(trackedPosition.height())
        cv2.rectangle(frame, (tracker_x, tracker_y), (tracker_x + tracker_w, tracker_y + tracker_h), (255,135,255), 3)
    cv2.imshow('Video', frame)
    if frame is None:
        break
    if k == 32:  # SPACE pressed
        img_name = "FaceDetect_webcam_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} saved!".format(img_name, frame))
        img_counter += 1
    if k == 27:
        break
