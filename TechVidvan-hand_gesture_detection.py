import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Initialize MediaPipe hands and load the gesture model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Load the model and gesture class names
try:
    model = load_model('mp_hand_gesture')
    print("Model loaded successfully.")
except Exception as e:
    print("Model loading failed:", e)

with open('gesture.names', 'r') as f:
    class_names = f.read().splitlines()

# Set up webcam capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame for a mirror effect
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hand landmarks
    result = hands.process(frame_rgb)

    # Initialize variables for storing predictions for each hand
    right_hand_prediction = None
    left_hand_prediction = None

    if result.multi_hand_landmarks:
        for hand_idx, (hand_landmarks, handedness) in enumerate(zip(result.multi_hand_landmarks, result.multi_handedness)):
            # Determine if the hand is left or right
            is_right_hand = handedness.classification[0].label == 'Right'

            # Collect landmarks for prediction
            landmarks = []
            for lm in hand_landmarks.landmark:
                lmx = int(lm.x * frame.shape[1])
                lmy = int(lm.y * frame.shape[0])
                landmarks.append([lmx, lmy])

            # Reshape landmarks as needed for the model and make predictions
            if landmarks:
                landmarks = np.array(landmarks).reshape(1, 21, 2)
                prediction = model.predict(landmarks)
                confidence = np.max(prediction)
                class_id = np.argmax(prediction)

                # Only display if confidence is above threshold
                if confidence > 0.8:
                    if is_right_hand:
                        right_hand_prediction = class_names[class_id]
                    else:
                        left_hand_prediction = class_names[class_id]

            # Draw landmarks on the frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display right hand prediction at the top and left hand prediction below
    if right_hand_prediction:
        cv2.putText(frame, f"Right: {right_hand_prediction}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)
    if left_hand_prediction:
        cv2.putText(frame, f"Left: {left_hand_prediction}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


