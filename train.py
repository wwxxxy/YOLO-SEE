from ultralytics import YOLO

model_yaml_path = r"yolosee.yaml"
data_yaml_path = r"VisDrone2019_DET.yaml"


if __name__ == "__main__":
    # 加载训练模型
    model = YOLO(model_yaml_path)

    # 训练模型
    results = model.train(
        data=data_yaml_path,
        epochs=300,
        batch=16,
        amp=False,
        name="YOLO-SEE",
    )
