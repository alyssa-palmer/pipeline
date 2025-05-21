import cv2
import numpy as np
import json
import os

VIDEO_PATH = r"videos\6s_c_mback.mp4"
FRAME_OUTPUT_DIR = "output/framesv2"
JSON_OUTPUT_PATH = "output/blobs/blobs.json"

os.makedirs(FRAME_OUTPUT_DIR, exist_ok=True)

def preprocess_frame(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply thresholding (tune this!)
    _, thresh = cv2.threshold(blurred, 75, 255, cv2.THRESH_BINARY)

    return thresh

def detect_blobs(thresh):
    # Connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)

    blobs = []
    for i in range(1, num_labels):  # skip background
        x, y = int(centroids[i][0]), int(centroids[i][1])
        area = stats[i, cv2.CC_STAT_AREA]

        # Optional: Filter small blobs
        if area < 50:
            continue

        blobs.append({
            "object_id": i,
            "x": x,
            "y": y,
            "area": int(area)
        })

    return blobs

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_id = 0
    output_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed during read.")
            break

        # Save original frame (optional for debugging)
        cv2.imwrite(f"{FRAME_OUTPUT_DIR}/frame_{frame_id:04d}.png", frame)

        thresh = preprocess_frame(frame)
        cv2.imwrite(f"{FRAME_OUTPUT_DIR}/thresh_{frame_id:04d}.png", thresh)
        blobs = detect_blobs(thresh)

        frame_info = {
            "frame_id": frame_id,
            "blobs": blobs
        }
        output_data.append(frame_info)

        frame_id += 1

    cap.release()

    # Save to JSON
    with open(JSON_OUTPUT_PATH, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Processed {frame_id} frames. Blob data saved to {JSON_OUTPUT_PATH}")

if __name__ == "__main__":
    main()
