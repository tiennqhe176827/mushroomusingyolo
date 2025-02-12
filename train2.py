from torch.utils.data._utils import worker
from ultralytics import YOLO

if __name__ == '__main__':
    # Load model
    model = YOLO("yolo11n.pt")

    # Train using GPU
    model.train(data="C:/Users/nqt00/OneDrive/Desktop/mushroomRealizer/data.yaml",
                epochs=10, imgsz=640, device="cuda")  # hoặc device=0 nếu có nhiều GPU
