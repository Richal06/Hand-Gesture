import pickle
import cv2
import mediapipe as mp
import numpy as np
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'model.p')
if os.path.exists(model_path):
    model_dict = pickle.load(open(model_path, 'rb'))
    model = model_dict['model']
else:
    print(f"Error: '{model_path}' not found.")
    exit()

#<------------------------ Change the camera index to the correct one--------------------------->
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'A', 1: 'B', 2: 'L'}

while True:

    data_aux = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                data_aux.append(x)
                data_aux.append(y)

        # <--------------------Normalize the hand landmarks------------------------>
        data_aux = [(x - min(data_aux[0::2])) / (max(data_aux[0::2]) - min(data_aux[0::2])) for x in data_aux]

        # <------------------------Ensure data_aux has 42 features------------------------>
        while len(data_aux) < 42:
            data_aux.extend([0, 0])

        prediction = model.predict([np.asarray(data_aux[:42])])

        predicted_character = labels_dict[int(prediction[0])]

        cv2.putText(frame, predicted_character, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()