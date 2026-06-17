from function import *
import os
import cv2

for action in actions:
    os.makedirs(os.path.join(DATA_PATH, action), exist_ok=True)

with mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

    for action in actions:

        for sequence in range(no_sequences):

            path = f"Image/{action}/{sequence}.png"

            frame = cv2.imread(path)

            if frame is None:
                print("Missing:", path)
                continue

            image, results = mediapipe_detection(frame, hands)

            draw_styled_landmarks(image, results)

            keypoints = extract_keypoints(results)

            np.save(
                os.path.join(
                    DATA_PATH,
                    action,
                    f"{sequence}.npy"
                ),
                keypoints
            )

            print("Saved:", action, sequence)

cv2.destroyAllWindows()