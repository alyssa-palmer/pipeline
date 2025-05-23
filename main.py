import cv2
import numpy as np
import json
import os

VIDEO_PATH = "videos/5s_c_mback.mp4"
FRAME_OUTPUT_DIR = "output/frames"
JSON_OUTPUT_PATH = "output/blobs/blobs.json"

os.makedirs(FRAME_OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(JSON_OUTPUT_PATH), exist_ok=True)

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
            
        # Seat position logic using raw y value
        if y > 700:
            seat_position = "front"
        else:
            seat_position = "back"

        blobs.append({
            "object_id": i,
            "x": x,
            "y": y,
            "area": int(area),
            "seat_position": seat_position
        })

    return blobs

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_id = 0
    output_data = []
    previous_positions = {}

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

        updated_blobs = []
        for blob in blobs:
            obj_id = blob["object_id"]
            x, y = blob["x"], blob ["y"]

            # Save previous position if available
            if obj_id in previous_positions:
                prev_x, prev_y = previous_positions[obj_id]
                blob["prev_x"] = prev_x
                blob["prev_y"] = prev_y
            else:
                blob["prev_x"] = None
                blob["prev_y"] = None
            
            # Update for next frame
            previous_positions[obj_id] = (x, y)
            updated_blobs.append(blob)

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
