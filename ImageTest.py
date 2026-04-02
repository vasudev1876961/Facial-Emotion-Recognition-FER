import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(BASE_DIR, "faces-small.jpg")

model = load_model("best_model.h5")

labels_dict = {
    0:'Angry',1:'Disgust',2:'Fear',
    3:'Happy',4:'Neutral',5:'Sad',6:'Surprise'
}

faceDetect = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

frame = cv2.imread(image_path)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

faces = faceDetect.detectMultiScale(gray, 1.3, 5)

for x,y,w,h in faces:
    face = gray[y:y+h, x:x+w]
    face = cv2.resize(face, (48,48))
    face = face / 255.0
    face = face.reshape(1,48,48,1)

    result = model.predict(face, verbose=0)
    label = np.argmax(result)
    confidence = np.max(result)

    text = f"{labels_dict[label]} ({confidence*100:.1f}%)"

    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.putText(frame, text, (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255,255,255), 2)

cv2.imshow("Image Emotion", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()