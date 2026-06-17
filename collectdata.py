import cv2
import os

for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    os.makedirs(f"Image/{letter}", exist_ok=True)

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    roi = frame[40:400, 0:300]

    cv2.rectangle(frame, (0,40), (300,400), (255,255,255), 2)

    cv2.imshow("Camera", frame)
    cv2.imshow("ROI", roi)

    key = cv2.waitKey(1)

    if key == ord('a'):
        break

    if ord('a') <= key <= ord('z'):

        letter = chr(key).upper()

        count = len(os.listdir(f"Image/{letter}"))

        cv2.imwrite(
            f"Image/{letter}/{count}.png",
            roi
        )

        print(f"Saved {letter}")

cap.release()
cv2.destroyAllWindows()