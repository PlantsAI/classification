import argparse
from ultralytics import YOLO


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training script for YOLOv8 classification model on PlantsAI")
    parser.add_argument('--dataset', type=str, default='mnist160', help='dataset path')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--weights', type=str, default='yolov8m-cls.pt', help='initial weights path')
    args = parser.parse_args()
    print(args)

    # Load a model
    model = YOLO(args.weights)

    # Train the model
    model.train(data=args.dataset, epochs=args.epochs)

    # Evaluate the model
    metrics = model.val()  # no arguments needed, dataset and settings remembered
    print(metrics.top1)   # top1 accuracy
