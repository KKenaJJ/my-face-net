import cv2
import pickle
import face_recognition
import numpy as np


face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
ids = pickle.loads(open('face_encodings', 'rb').read())
known_encodings = ids['encodings']
known_names = ids['names']

vidcap = cv2.VideoCapture(0)

process_this_frame = True

while(True):
    _, frame = vidcap.read()
    frame_small = cv2.resize(frame, (0, 0), fx = 0.25, fy = 0.25)
    img_rgb_small = frame_small[:, :, ::-1]

    if process_this_frame:
        boxes = face_recognition.face_locations(img_rgb_small)
        encodings = face_recognition.face_encodings(img_rgb_small, boxes)


        names = []
        for  encoding in encodings:
            matches = face_recognition.compare_faces(known_encodings, encoding)
            name = 'Unknown'

            face_distances = face_recognition.face_distance(known_encodings, encoding)
            least_distance_idx = np.argmin(face_distances)
            if matches[least_distance_idx]:
                name = known_names[least_distance_idx]


            names.append(name)
    process_this_frame =  not process_this_frame

    for (y, endx, endy, x), name in zip(boxes, names):
        y *= 4
        endx *= 4
        endy *= 4
        x *= 4

        cv2.rectangle(frame, (x, y), (endx, endy), (0, 0, 255), 2)
        cv2.rectangle(frame, (x, endy), (endx, endy + 40), (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, name, (x + 6, endy + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


    cv2.imshow('Video', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

vidcap.release()
cv2.destroyAllWindows()