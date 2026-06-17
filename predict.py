import cv2
import numpy as np
from tensorflow.keras.models import load_model
from function import *

model = load_model("model.h5")

actions = np.array([
    'A','B','C','D','E','F','G','H',
    'I','J','K','L','M','N','O','P',
    'Q','R','S','T','U','V','W','X',
    'Y','Z'
])

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        image, results = mediapipe_detection(frame, hands)

        draw_styled_landmarks(image, results)

        keypoints = extract_keypoints(results)

        prediction = model.predict(
            np.expand_dims(keypoints, axis=0),
            verbose=0
        )[0]

        action = actions[np.argmax(prediction)]

        cv2.putText(
            image,
            f"Prediction : {action}",
            (10,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

        cv2.imshow("Prediction", image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()