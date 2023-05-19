import argparse
from ultralytics import YOLO


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Export to ONNX script for YOLOv8 classification model on PlantsAI dataset")
    parser.add_argument('--weights', type=str, default='runs/classify/train/weights/best.pt', help='initial weights path')
    args = parser.parse_args()
    print(args)

    # Load a model
    model = YOLO(args.weights)

    # Export the model
    model.export(format='onnx')
