from ultralytics import YOLO
import cv2
import torch
import sys
import csv
import os

# Constants
START_FRAME = 0
END_FRAME = 400 # Set to 0 for no limit
INPUT_VIDEO = "input.mp4"
OUTPUT_VIDEO = "output_with_detections.mp4"
OUTPUT_CSV = "player_positions.csv"

# Load the model
model = YOLO("yolo11m-pose.pt")

def process_frame(model, frame, frame_number, csv_writer):
    # Predict with the model
    results = model.predict(frame, save=False)[0]

    player_positions = []
    # Process each detection
    if len(results) > 0:
        # Get the bounding boxes
        boxes = results.boxes
        keypoints = results.keypoints.data

        # Sort boxes by x-coordinate to determine left (player1) and right (player2) players
        box_coords = []
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            box_coords.append((center_x, center_y, x1, y1, x2, y2))

        # Sort by x-coordinate
        box_coords.sort(key=lambda x: x[0])

        # Draw and record positions for leftmost and rightmost players only
        if len(box_coords) >= 2:
            # Get leftmost and rightmost players
            left_player = box_coords[0]
            right_player = box_coords[-1]
            players_to_draw = [left_player, right_player]

        for i, (center_x, center_y, x1, y1, x2, y2) in enumerate(players_to_draw):
            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                          (0, 255, 0), 2)
            player_positions.extend([center_x, center_y])

            # Draw keypoints for this player
            player_keypoints = keypoints[i]
            for kpt in player_keypoints:
                x, y = int(kpt[0]), int(kpt[1])
                cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

    # Ensure we always write a row even if players are not detected
    while len(player_positions) < 4:
        player_positions.extend([0, 0])  # Fill missing players with (0,0)

    # Write to CSV: frame, player1_x, player1_y, player2_x, player2_y
    csv_writer.writerow([frame_number] + player_positions)

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

    # Set up CSV writer
    csv_file = open(OUTPUT_CSV, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['frame', 'player1_position_x', 'player1_position_y',
                        'player2_position_x', 'player2_position_y'])

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
        processed_frame = process_frame(model, frame, current_frame, csv_writer)

        # Write the frame
        out.write(processed_frame)

        # Display progress
        print(f"Processing frame {current_frame}/{end_frame}", end='\r')
        current_frame += 1

    # Clean up
    cap.release()
    out.release()
    csv_file.close()
    cv2.destroyAllWindows()
    print(f"\nProcessing complete. Output saved to {OUTPUT_VIDEO}")
    print(f"Player positions saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()