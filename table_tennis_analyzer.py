from ultralytics import YOLO
import cv2
import torch
import sys

# Load the model
model = YOLO("yolov8n-pose.pt")
image_path = "table-tennis.png"

def process_image(model, image_path):
    # Read the image from the path
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to open image {image_path}")
        sys.exit(1)

    # Print image dimensions for debugging
    height, width = image.shape[:2]
    print(f"Image dimensions: {width}x{height}")

    # Ensure the image is in a reasonable size for processing
    max_dimension = 1280
    if width > max_dimension or height > max_dimension:
        scale = max_dimension / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height))
        print(f"Resized image to: {new_width}x{new_height}")

    # Predict with the model
    results = model.predict(image, save=False)[0]

    # Process each detection
    for result in results:
        # Get the bounding boxes
        boxes = result.boxes
        # Get the keypoints
        keypoints = result.keypoints.data

        for box, kpts in zip(boxes, keypoints):
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            # Draw bounding box
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # Draw keypoints
            for kpt in kpts:
                x, y = int(kpt[0]), int(kpt[1])
                cv2.circle(image, (x, y), 3, (0, 0, 255), -1)
    return image

def main():
    if len(sys.argv) < 2:
        print("Usage: python table_tennis_analyzer.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    result_image = process_image(model, image_path)

    # Display the resulting image.
    cv2.imshow("Detected Poses", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()