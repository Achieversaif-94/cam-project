import numpy as np

def extract_features(hand_landmarks, image_width, image_height):
    """
    Convert 21 MediaPipe landmarks (new Tasks API) into a 42-dim normalized vector.
    """
    coords = []
    for lm in hand_landmarks:
        x = lm.x * image_width
        y = lm.y * image_height
        coords.append([x, y])

    coords = np.array(coords, dtype=np.float32)  # (21, 2)

    # 1. Translation: center on wrist
    wrist = coords[0].copy()
    coords -= wrist

    # 2. Scale: divide by bounding-box diagonal
    min_xy = coords.min(axis=0)
    max_xy = coords.max(axis=0)
    span = max_xy - min_xy
    scale = np.linalg.norm(span)
    if scale == 0:
        scale = 1.0
    coords /= scale

    return coords.flatten()  # shape: (42,)