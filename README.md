\# 🖐️ Sign Live — Real-Time ASL Fingerspelling Recognizer



A real-time Computer Vision app that translates ASL fingerspelling into text using MediaPipe hand tracking and a custom-trained SVC classifier.



\*\*🔴 Live Web Demo:\*\* \[https://cam-project.streamlit.app/](https://cam-project.streamlit.app/)

\*\*💻 Local Real-Time Demo:\*\* Run `python src/predict.py` for the full webcam word-building experience.



\## 🏗️ Architecture

```text

Webcam Snapshot

&#x20;  ↓

MediaPipe HandLandmarker (21 landmarks)

&#x20;  ↓

Feature Extraction (Normalization: center on wrist, scale by bbox)

&#x20;  ↓

SVC Classifier (29 classes: A-Z, space, del, nothing)

&#x20;  ↓

Predicted Letter + Confidence

📊 Model Evaluation

Test Accuracy: 99.83% (SVC with RBF kernel).



Dataset: Kaggle ASL Alphabet (subset of 3,480 images, 120 per class).



Features: 42-dimensional normalized vector (21 x,y coordinates).



🧠 Design Decisions

Landmarks over Raw Pixels: Instead of training a heavy CNN on raw images, we extract 21 hand landmarks. This makes the model <5MB, runs at 30 FPS on a CPU, and is invariant to background noise.



Normalization is Non-Negotiable: We subtract the wrist coordinate and divide by the bounding box size. Without this, the model would learn "hand position in frame" instead of "hand shape."



Stability Filter: In the live script (predict.py), a letter must be held for 15 consecutive frames (>90% confidence) before being added to the word builder, preventing flickering predictions.



⚠️ Limitations (Honest Caveats)

Static vs. Dynamic Letters: The model is trained on static images. Letters like 'J' and 'Z' require motion in real ASL, so the model only recognizes their static starting poses.



Domain Gap: The model is trained on the Kaggle dataset (studio lighting, specific skin tones). Live webcam accuracy may drop in poor lighting or with different backgrounds.



Fingerspelling Only: This translates individual letters, not full ASL grammar or word signs.

