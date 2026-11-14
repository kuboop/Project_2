import cv2
import mediapipe as mp
import joblib

model = joblib.load("hold_classifier_2.pkl")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

cap = cv2.VideoCapture(0)

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

            prediction = model.predict([row])[0]

            cv2.putText(frame, prediction, (10,50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0,255,0), 2)

    cv2.imshow("Hold detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()