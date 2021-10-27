import cv2
import pickle
import face_recognition
import numpy as np
from datetime import datetime
import pandas as pd


def mark_attendance(name):
    attendance = pd.read_csv('attendance.csv', index_col = [0])
    nameList = []
    if attendance['Name'].size:
        for n in attendance['Name'].values:
            nameList.append(n)

    else:
        nameList = []

    now = datetime.now()
    t = now.time()
    period1 = now.replace(now.year, now.month, now.day, 7, 45, 0)
    period2 = now.replace(now.year, now.month, now.day, 8, 35, 0)
    period3 = now.replace(now.year, now.month, now.day, 9, 25, 0)
    period4 = now.replace(now.year, now.month, now.day, 10, 15, 0)
    period5 = now.replace(now.year, now.month, now.day, 11, 5, 0)
    period6 = now.replace(now.year, now.month, now.day, 12, 49, 0)
    period7 = now.replace(now.year, now.month, now.day, 13, 39, 0)
    period8 = now.replace(now.year, now.month, now.day, 14, 29, 0)
    dayend = now.replace(now.year, now.month, now.day, 15, 15, 0)

    nameidx = 0
    for nameidx in range(len(attendance.index)):
        if attendance.loc[nameidx]['Name'] == name:
            continue
        else:
            nameidx += 1
    if name not in nameList:
        date_str = now.strftime('%H:%M:%S')
        nameList.append(name)
        attendance.loc[nameidx] = [name, date_str, 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']

    if name in nameList:
        if period1 <= now < period2:
            if not attendance.loc[nameidx].values[2] == 'X':
                attendance.loc[nameidx].values[2] = 'X'

        if period2 <= now < period3:
            if not attendance.loc[nameidx].values[3] == 'X':
                attendance.loc[nameidx].values[3] = 'X'

        if period3 <= now < period4:
            if not attendance.loc[nameidx].values[4] == 'X':
                attendance.loc[nameidx].values[4] = 'X'

        if period4 <= now < period5:
            if not attendance.loc[nameidx].values[5] == 'X':
                attendance.loc[nameidx].values[5] = 'X'

        if period5 <= now < period6:
            if not attendance.loc[nameidx].values[6] == 'X':
                attendance.loc[nameidx].values[6] = 'X'

        if period6 <= now < period7:
            if not attendance.loc[nameidx].values[7] == 'X':
                attendance.loc[nameidx].values[7] = 'X'

        if period7 <= now < period8:
            if not attendance.loc[nameidx].values[8] == 'X':
                attendance.loc[nameidx].values[8] = 'X'

        if period8 <= now < dayend:
            if not attendance.loc[nameidx].values[9] == 'X':
                attendance.loc[nameidx].values[9] = 'X'
    attendance.to_csv('attendance.csv')


def back(*args):
    pass

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
ids = pickle.loads(open('face_encodings', 'rb').read())
known_encodings = ids['encodings']
known_names = ids['names']

vidcap = cv2.VideoCapture(0)
# vidcap = cv2.VideoCapture(2)

process_this_frame = True

while (True):
    _, frame = vidcap.read()
    frame_small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    img_rgb_small = frame_small[:, :, ::-1]

    if process_this_frame:
        boxes = face_recognition.face_locations(img_rgb_small)
        encodings = face_recognition.face_encodings(img_rgb_small, boxes)

        names = []
        for encoding in encodings:
            matches = face_recognition.compare_faces(known_encodings, encoding)
            name = 'Unknown'

            face_distances = face_recognition.face_distance(known_encodings, encoding)
            least_distance_idx = np.argmin(face_distances)
            if matches[least_distance_idx]:
                name = known_names[least_distance_idx]
                mark_attendance(name)
            if name == 'Unknown':
                pass

            names.append(name)

    process_this_frame = not process_this_frame

    for (y, endx, endy, x), name in zip(boxes, names):
        y *= 4
        endx *= 4
        endy *= 4
        x *= 4

        cv2.rectangle(frame, (x, y), (endx, endy), (0, 0, 255), 2)





        cv2.createButton("Back",back,None,cv2.QT_PUSH_BUTTON,1)
        cv2.rectangle(frame, (x, endy), (endx, endy + 40), (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, name, (x + 6, endy + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Video', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

vidcap.release()
cv2.destroyAllWindows()