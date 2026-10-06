import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

MODEL_PATH = "model_single_final.h5"

CLASSES = ["C", "I", "L", "O", "U", "V"]

CONFIDENCE_THRESHOLD = 0.60

print("Loading single-hand model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")
print("Classes:", CLASSES)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


def normalize_landmarks(hand_landmarks):

    points = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
        dtype=np.float32
    )

    # Same normalization used during model training
    mean = points.mean(axis=0)
    points = points - mean

    span_x = np.ptp(points[:, 0])
    span_y = np.ptp(points[:, 1])

    span = max(span_x, span_y, 1e-6)

    points = points / span

    # Center again
    points = points - points.mean(axis=0)

    return points.flatten()


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("Webcam started!")
print("Press Q to quit.")

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not read webcam frame.")
        break

    # Mirror camera
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    prediction_text = "No hand detected"

    if results.multi_hand_landmarks:

        hand_landmarks = results.multi_hand_landmarks[0]

        # Draw hand landmarks
        mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

        # Extract 63 landmarks
        features = normalize_landmarks(hand_landmarks)

        input_data = np.expand_dims(features, axis=0)

        # Predict
        prediction = model.predict(
            input_data,
            verbose=0
        )

        index = int(np.argmax(prediction))

        confidence = float(prediction[0][index])

        if confidence >= CONFIDENCE_THRESHOLD:

            label = CLASSES[index]

            prediction_text = (
                f"{label} ({confidence * 100:.1f}%)"
            )

        else:

            prediction_text = (
                f"Uncertain ({confidence * 100:.1f}%)"
            )

    # Display prediction
    cv2.rectangle(
        frame,
        (10, 10),
        (420, 85),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        "ISL Prediction:",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        prediction_text,
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "Indian Sign Language Recognition",
        frame
    )

    # Press Q to quit
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
hands.close()

print("Webcam closed.")