from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")  # load an official model
model = YOLO("runs/detect/train/weights/best.pt")  # load a custom model

# Predict with the model
results = model(r"C:\Users\nqt00\OneDrive\Desktop\OSK.jpg")  # predict on an image