from ultralytics import YOLO
import cv2
import torch
import sys

# Constants
START_FRAME = 0
END_FRAME = 200
INPUT_VIDEO = "input.mp4"
OUTPUT_VIDEO = "output_with_detections.mp4"

# Load the model
model = YOLO("yolo11n-pose.pt")


def process_frame(model, frame):
    # Predict with the model
    results = model.predict(frame, save=False)[0]

    # Process each detection
    for result in results:
        # Get the bounding boxes and keypoints
        boxes = result.boxes
        keypoints = result.keypoints.data

        for box, kpts in zip(boxes, keypoints):
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # Draw keypoints
            for kpt in kpts:
                x, y = int(kpt[0]), int(kpt[1])
                cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

    return frame


def main():
    # Open the video file
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print(f"Error: Could not open video file {INPUT_VIDEO}")
        sys.exit(1)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (frame_width, frame_height))

    # Determine end frame
    end_frame = total_frames if END_FRAME == 0 else min(END_FRAME, total_frames)

    # Set starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)
    current_frame = START_FRAME

    while cap.isOpened() and current_frame < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        processed_frame = process_frame(model, frame)

        # Write the frame
        out.write(processed_frame)

        # Display progress
        print(f"Processing frame {current_frame}/{end_frame}", end='\r')
        current_frame += 1

    # Clean up
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\nProcessing complete. Output saved to {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()