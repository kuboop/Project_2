import cv2
import mediapipe as mp
import csv

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

label = input("Zadaj label (thumb, palm, fist, index): ")

with open("dataset_hold_2.csv", "a", newline="") as f:
    writer = csv.writer(f)

    while True:
        ret, frame = cap.read()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                row = []
                for lm in hand.landmark:
                    row.append(lm.x)
                    row.append(lm.y)

                row.append(label)
                writer.writerow(row)

                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Zber dat", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
