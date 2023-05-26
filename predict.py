import argparse
from ultralytics import YOLO


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict script for YOLOv8 classification model on PlantsAI")
    parser.add_argument('--input', type=str, default='io/input/image1.jpg', help='input image path')
    parser.add_argument('--weights', type=str, default='runs/classify/train/weights/best.pt', help='initial weights path')
    parser.add_argument('--image-size', type=int, default=224, help='size of input image')
    args = parser.parse_args()
    print(args)

    # Load a model
    model = YOLO(args.weights)

    # Predict with the model
    results = model(source=args.input, imgsz=[args.image_size, args.image_size])  # predict on an image
    
    for result in results:
        index = result.probs.argmax()
        class_name = result.names[int(index)]
        print(class_name)
