from ultralytics import YOLO

if __name__ == "__main__":
    pth_path = r"/root/autodl-fs/YOLO-SEE/runs/detect/000/weights/best.pt"

    test_path = r"/root/autodl-fs/YOLO-SEE/test"
    # Load a model
    model = YOLO(pth_path)  # load a custom model

    # Predict with the model
    results = model(
        test_path,
        save=True,
        imgsz=640,
        batch=16,
        name="YOLO-SEE",
    )
