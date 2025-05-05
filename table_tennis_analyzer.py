import os
from ultralytics import YOLO
import cv2
import torch
import sys
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm

# Constants
START_FRAME = 0
END_FRAME = 600 # Set to 0 for no limit
INPUT_VIDEO = "input.mp4"
OUTPUT_VIDEO = "output_with_detections.mp4"
OUTPUT_CSV = "player_positions.csv"

# Load the model
model = YOLO("yolo11l-pose.pt")

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

            # Define skeleton connections (modify indices if your model has a different keypoint format)
            SKELETON = [
            (5, 6),  # shoulders
            (5, 7),  # left arm: shoulder -> elbow
            (7, 9),  # left arm: elbow -> wrist
            (6, 8),  # right arm: shoulder -> elbow
            (8, 10),  # right arm: elbow -> wrist
            (5, 11),  # left torso: shoulder -> hip
            (6, 12),  # right torso: shoulder -> hip
            (11, 12),  # hips
            (11, 13),  # left leg: hip -> knee
            (13, 15),  # left leg: knee -> ankle
            (12, 14),  # right leg: hip -> knee
            (14, 16),  # right leg: knee -> ankle
            (0, 1),  # face connection example
            (1, 2),
            (0, 3),
            (3, 4)
            ]
            skeleton_color = (0, 255, 255)  # Yellow color for skeleton lines
            line_thickness = 2

            # Draw lines connecting keypoints based on the SKELETON definition
            for (p1, p2) in SKELETON:
                # Check if the indices are within range (if your model has a different number of keypoints)
                if p1 < len(player_keypoints) and p2 < len(player_keypoints):
                    x1_kpt, y1_kpt = int(player_keypoints[p1][0]), int(player_keypoints[p1][1])
                    x2_kpt, y2_kpt = int(player_keypoints[p2][0]), int(player_keypoints[p2][1])

                    # Optional: Skip drawing if coordinates are at (0,0) (could indicate a missing detection)
                    if (x1_kpt, y1_kpt) == (0, 0) or (x2_kpt, y2_kpt) == (0, 0):
                        continue
                    cv2.line(frame, (x1_kpt, y1_kpt), (x2_kpt, y2_kpt), skeleton_color, line_thickness)

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

    df = pd.read_csv('player_positions.csv') # Read the CSV file

    # Create a heatmap image from the first frame of the video
    cap2 = cv2.VideoCapture(INPUT_VIDEO)
    ret, first_frame = cap2.read()
    cap2.release()

    if not ret:
        print("Error: Could not read the first frame from video.")
        sys.exit(1)

    first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    img_height, img_width, _ = first_frame_rgb.shape

    # collecting player positions data
    x_positions = df['player1_position_x'].tolist() + df['player2_position_x'].tolist()
    y_positions = df['player1_position_y'].tolist() + df['player2_position_y'].tolist()

    plt.figure(figsize=(12, 6)) # Set the figure size
    plt.imshow(first_frame_rgb, extent=[0, img_width, img_height, 0])  # coordinate origin is top-left
    plt.axis('image') # remove unrellevant boarders and labels
    plt.gca().set_aspect('equal', adjustable='box') #adjust width and height


    #draw the heatmap
    sns.kdeplot(
        x=x_positions,
        y=y_positions,
        fill=True,
        cmap=cm.jet,
        bw_adjust= 1,
        alpha=0.6,
        levels=100,
        thresh=0.05
    )

    # set boarder, labels and colors
    plt.title('Combined Player Position Heatmap')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.colorbar(label='Density')

    # save the heatmap photo
    plt.savefig('player_position_heatmap.png', dpi=300, bbox_inches='tight', pad_inches=0.5)

    # Clean up
    cap.release()
    out.release()
    csv_file.close()
    cv2.destroyAllWindows()
    print(f"\nProcessing complete. Output saved to {OUTPUT_VIDEO}")
    print(f"Player positions saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()