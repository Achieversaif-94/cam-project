import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import joblib
from src.features import extract_features

# --- Page Config ---
st.set_page_config(page_title="Sign Live", page_icon="🖐️", layout="wide")

# --- Load Models (Cached so it's fast) ---
@st.cache_resource
def load_ai_models():
    # Load our SVC model
    bundle = joblib.load("models/asl_landmark_model.joblib")
    model = bundle["model"]
    
    # Load MediaPipe
    base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
    detector = vision.HandLandmarker.create_from_options(options)
    
    return model, detector

model, detector = load_ai_models()

st.title("🖐️ Sign Live — ASL Fingerspelling Recognizer")
st.markdown("Take a photo of your hand to instantly translate ASL fingerspelling into text!")

tab1, tab2, tab3 = st.tabs(["📸 Live Snapshot", "📊 Model Card", "⚙️ How It Works"])

with tab1:
    st.header("Try it out!")
    st.info("Hold your hand up to the camera and make an ASL letter. Ensure good lighting!")
    
    # Streamlit camera input
    img_file = st.camera_input("Take a picture")
    
    if img_file is not None:
        # Convert to OpenCV format
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Streamlit flips images by default sometimes, ensure it's RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape
        
        # Run MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)
        
        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            
            # Draw points on the image for visual feedback
            for landmark in hand:
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            
            # Predict
            vec = extract_features(hand, w, h).reshape(1, -1)
            proba = model.predict_proba(vec)[0]
            idx = int(np.argmax(proba))
            label = model.classes_[idx]
            confidence = float(proba[idx])
            
            # Show Results
            st.success(f"### Prediction: **{label.upper()}** (Confidence: {confidence*100:.1f}%)")
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Detected Landmarks", use_column_width=True)
        else:
            st.error("No hand detected! Make sure your hand is fully in the frame.")

with tab2:
    st.header("Model Evaluation")
    st.write("This model was trained on the Kaggle ASL Alphabet Dataset.")
    st.metric(label="Test Accuracy (SVC)", value="99.83%")
    st.write("""
    **Architecture details:**
    * **Input:** 42 normalized coordinates (21 landmarks × x,y).
    * **Classifier:** Support Vector Classifier (RBF Kernel).
    * **Training Subset:** 3,480 images (120 per class).
    * **Why landmarks?** Raw pixels are heavy and sensitive to background. Normalized landmarks focus purely on hand geometry, allowing the model to run efficiently on a CPU.
    """)

with tab3:
    st.header("The Pipeline")
    st.code("""
Webcam → MediaPipe HandLandmarker (21 dots)
       → Normalization (wrist-center, bounding-box scale)
       → SVC Model
       → Label + Confidence
    """, language="text")
    st.markdown("""
    ### Design Decisions
    * **Normalization:** We subtract the wrist coordinate and divide by the bounding box size. This ensures the AI recognizes the letter shape regardless of where your hand is in the frame or how close you are to the camera.
    * **Stability Filter (Local App):** In the `predict.py` script, a prediction must be held for 15 frames before being added to the word, preventing flickering.
    """)