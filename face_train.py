import pickle
import face_recognition
from imutils import paths
import os

img_paths = list(paths.list_images('images'))
knownEncodings = []
knownLabels = []

for (i, img_path) in enumerate(img_paths):
    name = img_path.split(os.path.sep)[-2]
    img = face_recognition.load_image_file(img_path)
    boxes = face_recognition.face_locations(img, model = 'hog')
    encodings = face_recognition.face_encodings(img, boxes)

    for encoding in encodings:
        knownEncodings.append(encoding)
        knownLabels.append(name)

ids = {'encodings': knownEncodings, 'names': knownLabels}
with open ('face_encodings', 'wb') as f:
    f.write(pickle.dumps(ids))
    