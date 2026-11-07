import os
import cv2
import joblib
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from features import extract_features

# --- Config ---
DATA_DIR = "data/asl_alphabet_train"
MODEL_DIR = "models"
SAMPLE_PER_CLASS = 120  # We only take 120 images per letter to keep it fast!
MODEL_PATH = "hand_landmarker.task"

os.makedirs(MODEL_DIR, exist_ok=True)

def load_dataset():
    X, y = [], []
    
    # Initialize New MediaPipe API
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
    detector = vision.HandLandmarker.create_from_options(options)

    classes = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    print(f"Classes to train on: {classes}")

    for label in classes:
        folder = os.path.join(DATA_DIR, label)
        files = os.listdir(folder)[:SAMPLE_PER_CLASS]
        
        for fname in tqdm(files, desc=f"Processing {label}"):
            path = os.path.join(folder, fname)
            img = cv2.imread(path)
            if img is None:
                continue
                
            h, w, _ = img.shape
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Convert to MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_image)
            
            # If a hand is found in the image, extract features
            if result.hand_landmarks:
                hand = result.hand_landmarks[0]
                vec = extract_features(hand, w, h)
                X.append(vec)
                y.append(label)

    return np.array(X), np.array(y)

def train():
    X, y = load_dataset()
    print(f"\nLoaded {len(X)} samples, {X.shape[1]} features each.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "SVC": SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42),
    }

    best_name, best_model, best_acc = None, None, 0.0

    for name, model in models.items():
        print(f"\nTraining {name}... (this might take a minute)")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"{name} accuracy: {acc:.4f}")
        
        if acc > best_acc:
            best_name, best_model, best_acc = name, model, acc

    save_path = os.path.join(MODEL_DIR, "asl_landmark_model.joblib")
    joblib.dump({"model": best_model, "model_name": best_name}, save_path)
    print(f"\n✅ Saved best model ({best_name}, {best_acc:.4f}) -> {save_path}")

if __name__ == "__main__":
    train()