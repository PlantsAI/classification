import argparse
from ultralytics import YOLO


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict script for YOLOv8 classification model on PlantsAI dataset")
    parser.add_argument('--input', type=str, default='io/input/image.jpg', help='input image path')
    parser.add_argument('--weights', type=str, default='runs/classify/train/weights/best.pt', help='initial weights path')
    args = parser.parse_args()
    print(args)

    # Load a model
    model = YOLO(args.weights)

    # Predict with the model
    results = model(args.input)  # predict on an image
    print(results[0])  # print results to screen
