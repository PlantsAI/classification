# PlantsAI Classification

This repository contains the code for training a YOLOv8 classification model on PlantsAI dataset and exporting it to ONNX. The model is then used for inference on a sample image. 

## Installation

```bash
pip install -r requirements.txt
```

## Run predict demo

The easiest way to get started is to use our trained model on your own images to perform classification. To run the demo, follow these steps: 
1. Download pretrained model from [Google Drive](#)
2. Run command below
```bash
python predict.py
```

## Dataset

Download PlantsAI dataset for training and evaluation from [Google Drive](#)


## Train and evaluate

Run command below to train and evaluate the model on PlantsAI dataset. .
```bash
python train.py
```
The model will be saved to `runs/classify/train/weights/best.pt`

## Export to ONNX

Run command below to export the model to ONNX format.
```bash
python export.py
```
The model will be saved to `runs/classify/train/weights/best.onnx`.
