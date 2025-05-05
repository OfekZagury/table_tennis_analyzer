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
import easyocr  # For OCR scoreboard reading
import re  # For parsing digits from OCR results

START_FRAME = 0
END_FRAME = 600  # Set to 0 for processing all frames in the video
INPUT_VIDEO = "input.mp4"
OUTPUT_VIDEO = "output_with_detections.mp4"
OUTPUT_CSV = "player_positions.csv"

# New constants for scoreboard reading and score plotting
SCOREBOARD_CSV = "player_scores.csv"
SCORE_READ_INTERVAL = 10  # Read scoreboard every 10 frames

class ScoreTracker:
    def __init__(self):
        self.last_valid_scores = (0, 0)
        self.score_history = []

    def update_scores(self, frame_num, new_scores):
        if all(0 <= score <= 41 for score in new_scores):
            self.last_valid_scores = new_scores
            self.score_history.append((frame_num, *new_scores))
        return self.last_valid_scores


class FrameValidator:
    def __init__(self):
        self.reference_positions = None
        self.reference_sizes = None

    def set_reference_frame(self, boxes, frame_width, frame_height):
        if len(boxes) >= 2:
            box_coords = []
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                center_x = (x1 + x2) / 2
                width = x2 - x1
                height = y2 - y1
                box_coords.append((center_x, y1, width, height))

            # Sort by x-coordinate
            box_coords.sort(key=lambda x: x[0])

            # Store reference positions as normalized coordinates
            self.reference_positions = [
                (box_coords[0][0] / frame_width, box_coords[0][1] / frame_height),  # Left player
                (box_coords[-1][0] / frame_width, box_coords[-1][1] / frame_height)  # Right player
            ]

            # Store reference sizes as normalized values
            self.reference_sizes = [
                (box_coords[0][2] / frame_width, box_coords[0][3] / frame_height),  # Left player
                (box_coords[-1][2] / frame_width, box_coords[-1][3] / frame_height)  # Right player
            ]
            return True
        return False

    def is_valid_frame(self, boxes, keypoints, frame_width, frame_height):
        # Basic detection checks
        if len(boxes) < 2:
            return False

        box_coords = []
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            box_coords.append((center_x, center_y, width, height))

        # Sort players by x-coordinate
        box_coords.sort(key=lambda x: x[0])
        p1 = box_coords[0]  # Leftmost player
        p2 = box_coords[-1]  # Rightmost player

        # Enhanced position and size checks
        constraints = {
            'min_horizontal_separation': 0.3,  # Players must be at least 40% of frame width apart
            'max_vertical_difference': 0.5,  # Players must be within 20% frame height of each other
            'valid_y_min': 0.1,  # Players must be between 20% and 80% of frame height
            'valid_y_max': 0.9,
            'min_player_height': 0.1,  # Player height must be between 15% and 60% of frame height
            'max_player_height': 0.8,
            'position_tolerance': 0.2,  # Players must be within 10% of reference positions
            'size_tolerance': 0.15  # Player size must be within 20% of reference size
        }

        # Check horizontal separation
        if abs(p2[0] - p1[0]) < frame_width * constraints['min_horizontal_separation']:
            return False

        # Check vertical alignment
        if abs(p2[1] - p1[1]) > frame_height * constraints['max_vertical_difference']:
            return False

        # Check vertical positions
        valid_y_min = frame_height * constraints['valid_y_min']
        valid_y_max = frame_height * constraints['valid_y_max']
        if not (valid_y_min < p1[1] < valid_y_max and valid_y_min < p2[1] < valid_y_max):
            return False

        # Check player sizes
        min_height = frame_height * constraints['min_player_height']
        max_height = frame_height * constraints['max_player_height']
        if not (min_height < p1[3] < max_height and min_height < p2[3] < max_height):
            return False

        # Compare with reference frame if available
        if self.reference_positions is not None:
            # Check position consistency
            current_positions = [
                (p1[0] / frame_width, p1[1] / frame_height),
                (p2[0] / frame_width, p2[1] / frame_height)
            ]

            for i in range(2):
                position_diff = (
                    abs(current_positions[i][0] - self.reference_positions[i][0]),
                    abs(current_positions[i][1] - self.reference_positions[i][1])
                )
                if position_diff[0] > constraints['position_tolerance'] or \
                        position_diff[1] > constraints['position_tolerance']:
                    return False

            # Check size consistency
            current_sizes = [
                (p1[2] / frame_width, p1[3] / frame_height),
                (p2[2] / frame_width, p2[3] / frame_height)
            ]

            for i in range(2):
                size_diff = (
                    abs(current_sizes[i][0] - self.reference_sizes[i][0]),
                    abs(current_sizes[i][1] - self.reference_sizes[i][1])
                )
                if size_diff[0] > constraints['size_tolerance'] or \
                        size_diff[1] > constraints['size_tolerance']:
                    return False

        return True


def get_scoreboard_region(frame_width, frame_height):
    # Bottom-left corner with adjusted dimensions
    x1 = 0
    y1 = int(frame_height * 0.75)  # Start from 75% down
    x2 = int(frame_width * 0.2)    # Use 20% of width
    y2 = frame_height              # Bottom of frame
    return (x1, y1, x2, y2)

# Load the pose detection model
model = YOLO("yolo11s-pose.pt")

# Initialize the EasyOCR reader; set gpu=True if you have a GPU available
ocr_reader = easyocr.Reader(['en'], gpu=False)


def read_scoreboard(frame, region, reader):
    try:
        x1, y1, x2, y2 = region
        scoreboard_crop = frame[y1:y2, x1:x2]

        # Image preprocessing for better OCR
        gray = cv2.cvtColor(scoreboard_crop, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)

        # Try different preprocessing methods
        results = []
        for img in [thresh, gray, scoreboard_crop]:
            ocr_result = reader.readtext(img, detail=0)
            if ocr_result:
                results.extend(ocr_result)

        print(f"OCR detected text: {results}")  # Debug print
        return results
    except Exception as e:
        print(f"Error reading scoreboard: {e}")
        return []



def parse_scoreboard_text(ocr_text_lines):
    def extract_score(text):
        # Clean text
        text = text.strip().replace('O', '0').replace('o', '0')
        text = ''.join(c for c in text if c.isdigit() or c in ':-')

        # Print debug info
        print(f"Cleaned text: {text}")

        # Try different score patterns
        score_patterns = [
            r'^(\d)-(\d)$',  # 1-0
            r'^(\d):(\d)$',  # 1:0
            r'^(\d)(\d)$',  # 10
            r'^(\d)$'  # 1
        ]

        for pattern in score_patterns:
            match = re.match(pattern, text)
            if match:
                try:
                    if len(match.groups()) == 2:
                        games, points = map(int, match.groups())
                    else:
                        num = int(text)
                        games = num // 10
                        points = num % 10

                    if 0 <= games <= 3 and 0 <= points <= 11:
                        score = games * 10 + points
                        print(f"Valid score found: {score}")
                        return score
                except:
                    continue
        return None

    scores = []
    for line in ocr_text_lines:
        if isinstance(line, str):
            score = extract_score(line)
            if score is not None:
                scores.append(score)

    print(f"Final scores: {scores}")  # Debug print
    return (scores[0], scores[1]) if len(scores) >= 2 else (0, 0)



def process_frame(model, frame, frame_number, csv_writer):
    frame_height, frame_width = frame.shape[:2]
    results = model.predict(frame, save=False)[0]
    boxes = results.boxes
    keypoints = results.keypoints.data

    player_positions = []
    if len(results) > 0:
        # Collect box coordinate information
        box_coords = []
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            box_coords.append((center_x, center_y, x1, y1, x2, y2))

        # Sort detections based on center x-coordinate to identify leftmost and rightmost players.
        box_coords.sort(key=lambda x: x[0])

        # Select leftmost and rightmost players.
        players_to_draw = [box_coords[0], box_coords[-1]]

        for i, (center_x, center_y, x1, y1, x2, y2) in enumerate(players_to_draw):
            # Draw bounding box for the player.
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                          (0, 255, 0), 2)  # Green box
            # Record the center position.
            player_positions.extend([center_x, center_y])

            # Draw keypoints for the current player.
            # NOTE: This indexing assumes that the ordering in keypoints matches boxes.
            player_keypoints = keypoints[i]
            for kpt in player_keypoints:
                x, y = int(kpt[0]), int(kpt[1])
                cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)  # Red circles

            # Define skeleton connections (update indices if model keypoint definitions differ)
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
                (0, 1),  # example face connection
                (1, 2),
                (0, 3),
                (3, 4)
            ]
            skeleton_color = (0, 255, 255)  # Yellow for skeleton lines
            line_thickness = 2

            # Draw skeleton lines connecting keypoints.
            for (p1, p2) in SKELETON:
                if p1 < len(player_keypoints) and p2 < len(player_keypoints):
                    x1_kpt, y1_kpt = int(player_keypoints[p1][0]), int(player_keypoints[p1][1])
                    x2_kpt, y2_kpt = int(player_keypoints[p2][0]), int(player_keypoints[p2][1])
                    # Skip if the keypoint is missing (0,0)
                    if (x1_kpt, y1_kpt) == (0, 0) or (x2_kpt, y2_kpt) == (0, 0):
                        continue
                    cv2.line(frame, (x1_kpt, y1_kpt), (x2_kpt, y2_kpt), skeleton_color, line_thickness)

    # Ensure row always has 4 coordinate values, even if players are not detected.
    while len(player_positions) < 4:
        player_positions.extend([0, 0])

    # Write current frame number and player positions to CSV.
    csv_writer.writerow([frame_number] + player_positions)

    return frame


def plot_scores(score_df, sample_rate=10):
    # Filter out invalid scores and sample data
    filtered_df = score_df[
                      (score_df['p1_calculated_score'] > 0) &
                      (score_df['p2_calculated_score'] > 0)
                      ].iloc[::sample_rate]

    plt.figure(figsize=(12, 6))
    plt.plot(filtered_df['frame'], filtered_df['p1_calculated_score'],
             label='Player 1', color='blue', marker='.', linestyle='-')
    plt.plot(filtered_df['frame'], filtered_df['p2_calculated_score'],
             label='Player 2', color='red', marker='.', linestyle='-')

    plt.title('Match Score Progression')
    plt.xlabel('Frame Number')
    plt.ylabel('Score (Games×10 + Points)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('score_progression.png', dpi=300)
    plt.close()



def main():
    # Open the video file.
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print(f"Error: Could not open video file {INPUT_VIDEO}")
        return

    score_tracker = ScoreTracker()

    # Get video properties.
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Set up video writer for output video with detections.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (frame_width, frame_height))

    # Set up CSV writer for player positions.
    csv_file = open(OUTPUT_CSV, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['frame', 'player1_position_x', 'player1_position_y',
                         'player2_position_x', 'player2_position_y'])

    # Set up CSV writer for scoreboard data.
    with open(SCOREBOARD_CSV, 'w', newline='') as score_file:
        score_writer = csv.writer(score_file)
        score_writer.writerow(['frame', 'p1_calculated_score', 'p2_calculated_score'])
    score_file = open(SCOREBOARD_CSV, 'a', newline='')
    score_writer = csv.writer(score_file)

    # Determine the ending frame.
    end_frame = total_frames if END_FRAME == 0 else min(END_FRAME, total_frames)

    # Set starting frame.
    cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)
    current_frame = START_FRAME

    # Initialize the frame validator
    frame_validator = FrameValidator()
    first_frame = True

    # Process each frame.
    while cap.isOpened() and current_frame < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        # Get model predictions
        results = model.predict(frame, save=False)[0]
        boxes = results.boxes
        keypoints = results.keypoints.data

        # Set reference frame from first valid detection
        if first_frame:
            if frame_validator.set_reference_frame(boxes, frame_width, frame_height):
                first_frame = False
                processed_frame = process_frame(model, frame, current_frame, csv_writer)
            else:
                # Write dummy values for invalid first frame
                csv_writer.writerow([current_frame, 0, 0, 0, 0])
                processed_frame = frame
            current_frame += 1
            out.write(processed_frame)
            continue

        # Use enhanced frame validation
        if frame_validator.is_valid_frame(boxes, keypoints, frame_width, frame_height):
            processed_frame = process_frame(model, frame, current_frame, csv_writer)
        else:
            # Write dummy values for invalid frames
            csv_writer.writerow([current_frame, 0, 0, 0, 0])
            processed_frame = frame


        if current_frame % SCORE_READ_INTERVAL == 0:
            try:
                frame_height, frame_width = frame.shape[:2]
                scoreboard_region = get_scoreboard_region(frame_width, frame_height)
                ocr_text_lines = read_scoreboard(frame, scoreboard_region, ocr_reader)
                if ocr_text_lines:  # Only process if OCR found text
                    new_scores = parse_scoreboard_text(ocr_text_lines)
                    current_scores = score_tracker.update_scores(current_frame, new_scores)
                    score_writer.writerow([current_frame, *current_scores])
                    score_file.flush()  # Force write to disk
            except Exception as e:
                print(f"Error processing score at frame {current_frame}: {e}")

        # Write the processed frame to the output video.
        out.write(processed_frame)

        # Display processing progress.
        print(f"Processing frame {current_frame}/{end_frame}", end='\r')
        current_frame += 1

    # Read the positions CSV to generate the positional heatmap.
    df = pd.read_csv(OUTPUT_CSV)

    # Create a heatmap image using the first frame as background.
    cap2 = cv2.VideoCapture(INPUT_VIDEO)
    ret, first_frame = cap2.read()
    cap2.release()

    if not ret:
        print("Error: Could not read the first frame from video.")
        sys.exit(1)

    # Convert first frame to RGB for displaying with matplotlib.
    first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    img_height, img_width, _ = first_frame_rgb.shape

    # Collect player positions data for both players.
    x_positions = df['player1_position_x'].tolist() + df['player2_position_x'].tolist()
    y_positions = df['player1_position_y'].tolist() + df['player2_position_y'].tolist()

    # Plot and save the positional heatmap.
    plt.figure(figsize=(12, 6), dpi=300)  # Figure size and resolution
    plt.imshow(first_frame_rgb, extent=[0, img_width, img_height, 0])  # Display background image with proper extent
    plt.axis('image')  # Use image axis for proper aspect ratio
    plt.gca().set_aspect('equal', adjustable='box')

    # Draw the kernel density estimate (KDE) heatmap with chosen parameters.
    sns.kdeplot(
        x=x_positions,
        y=y_positions,
        fill=True,
        cmap=cm.jet,
        bw_adjust=1,  # Adjust kernel bandwidth (tweak as needed)
        alpha=0.6,
        levels=100,
        thresh=0.05
    )

    plt.title('Combined Player Position Heatmap')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.colorbar(label='Density')
    plt.savefig('player_position_heatmap.png', dpi=300, bbox_inches='tight', pad_inches=0.5)

    # Plot the score progression
    try:
        score_df = pd.read_csv(SCOREBOARD_CSV)
        if not score_df.empty:
            plot_scores(score_df, sample_rate=30)
        else:
            print("Warning: No score data to plot")
    except Exception as e:
        print(f"Error plotting scores: {e}")

    # Clean up resources
    print("Cleaning up...")
    try:
        cap.release()
        out.release()
        csv_file.close()
        score_file.close()
    except Exception as e:
        print(f"Error during cleanup: {e}")

    print(f"\nProcessing complete.")
    print(f"Output video saved to {OUTPUT_VIDEO}")
    print(f"Player positions saved to {OUTPUT_CSV}")
    print(f"Score data saved to {SCOREBOARD_CSV}")


if __name__ == "__main__":
    main()
