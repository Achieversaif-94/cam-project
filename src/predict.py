import cv2
import joblib
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from features import extract_features

MODEL_PATH = "models/asl_landmark_model.joblib"
HAND_MODEL_PATH = "hand_landmarker.task"

def main():
    # Load our trained model
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    print(f"Loaded {bundle['model_name']} model with {len(model.classes_)} classes.")

    # Initialize New MediaPipe API
    base_options = python.BaseOptions(model_asset_path=HAND_MODEL_PATH)
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1, min_hand_detection_confidence=0.7)
    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)

    # --- Word Builder Variables ---
    current_word = ""
    stable_letter = None
    stable_count = 0
    STABILITY_THRESHOLD = 15  # Needs to be held for 15 frames
    CONFIDENCE_THRESHOLD = 0.90 # Must be 90% confident

    print("Press 'q' to quit. Hold a letter steady to build a word!")

    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect landmarks
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)

        label = "—"
        confidence = 0.0

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            
            # Draw green dots on the hand
            for landmark in hand:
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

            # Predict
            vec = extract_features(hand, w, h).reshape(1, -1)
            proba = model.predict_proba(vec)[0]
            idx = int(np.argmax(proba))
            label = model.classes_[idx]
            confidence = float(proba[idx])

            # --- Word Builder Logic ---
            if confidence > CONFIDENCE_THRESHOLD:
                if label == stable_letter:
                    stable_count += 1
                else:
                    stable_letter = label
                    stable_count = 0

                if stable_count == STABILITY_THRESHOLD:
                    if label == "space":
                        current_word += " "
                    elif label == "del":
                        current_word = current_word[:-1]
                    elif label != "nothing":
                        current_word += label
                    stable_count = 0  # Reset so we don't spam the letter
            else:
                stable_letter = None
                stable_count = 0

        # Draw UI on frame
        color = (0, 255, 0) if confidence > 0.85 else (0, 165, 255)
        cv2.putText(frame, f"{label} {confidence:.2f}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        cv2.putText(frame, f"Word: {current_word}", (10, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)

        cv2.imshow("Sign Live — Prediction", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()