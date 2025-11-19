import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import argparse
import os

def load_yolo_model(weights_path="yolov3.weights", cfg_path="yolov3.cfg", names_path="coco.names"):
    """
    Loads the YOLOv3 model from the specified files.
    """
    if not all(os.path.exists(p) for p in [weights_path, cfg_path, names_path]):
        print("Error: Model files not found. Please make sure 'yolov3.weights', 'yolov3.cfg', and 'coco.names' are in the same directory as the script.")
        return None, None, None

    net = cv2.dnn.readNet(weights_path, cfg_path)
    with open(names_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, classes, output_layers

def detect_objects(net, classes, output_layers, image_path, confidence_threshold=0.5):
    """
    Detects objects in an image using the loaded YOLO model.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image from path: {image_path}")
            return

        img = cv2.resize(img, None, fx=0.4, fy=0.4)
        height, width, channels = img.shape
    except Exception as e:
        print(f"An error occurred while loading the image: {e}")
        return

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing information on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.4)

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = (0, 255, 0)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label + " " + confidence, (x, y + 20), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
            print(f"Detected: {label} (Confidence: {confidence})")
            print(f"  - Width: {w} pixels")
            print(f"  - Height: {h} pixels")

        # Show the image with all detected objects
        cv2.imshow("Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No objects detected with the current confidence threshold.")

def detect_coins(image_path):
    """
    Detects coins in an image using Hough Circle Transform.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image from path: {image_path}")
            return
    except Exception as e:
        print(f"An error occurred while loading the image: {e}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Use GaussianBlur to reduce noise and improve circle detection
    gray = cv2.GaussianBlur(gray, (9, 9), 2)

    # The parameters of HoughCircles might need to be tuned for your specific images
    # dp: Inverse ratio of accumulator resolution to image resolution.
    # minDist: Minimum distance between the centers of the detected circles.
    # param1: Higher threshold for the Canny edge detector.
    # param2: Accumulator threshold for circle centers at the detection stage.
    # minRadius: Minimum circle radius.
    # maxRadius: Maximum circle radius.
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=50,
        param2=45,
        minRadius=120,
        maxRadius=400
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        print(f"Detected {len(circles[0, :])} coin(s).")
        for i in circles[0, :]:
            # Draw the outer circle
            cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # Draw the center of the circle
            cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
            print(f"  - Position: (x={i[0]}, y={i[1]}), Radius: {i[2]}")

        # Show the image with all detected objects
        cv2.imshow("Detected Coins", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No coins detected. You may need to adjust the detection parameters in the script for your specific image.")

def show_available_objects(classes):
    """
    Prints the list of objects the model can detect.
    """
    if not classes:
        print("Could not load class names.")
        return
    print("The current model can detect the following objects:")
    for obj in classes:
        print(f"- {obj}")

def main():
    parser = argparse.ArgumentParser(description="Object detection using YOLOv3 or coin detection using Hough Circles.")
    parser.add_argument("--show-objects", action="store_true", help="Show the list of detectable objects (for YOLO).")
    parser.add_argument("--confidence", type=float, default=0.5, help="Set the minimum detection confidence for YOLO (0.0 to 1.0).")
    parser.add_argument("--detect-coins", action="store_true", help="Use the specialized coin detector.")
    args = parser.parse_args()

    if args.detect_coins:
        try:
            root = tk.Tk()
            root.withdraw()
            file_path = filedialog.askopenfilename(title="Select an image file for coin detection", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
            if file_path:
                detect_coins(file_path)
        except Exception as e:
            print(f"An error occurred: {e}")
        return

    net, classes, output_layers = load_yolo_model()
    if not net:
        return

    if args.show_objects:
        show_available_objects(classes)
        return

    try:
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(title="Select an image file for object detection", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            detect_objects(net, classes, output_layers, file_path, args.confidence)
    except ImportError:
        print("Required libraries not found.")
        print("Please install them using:")
        print("pip install opencv-python numpy tkinter")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
