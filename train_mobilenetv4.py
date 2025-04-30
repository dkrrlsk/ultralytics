from ultralytics import YOLO

# 两阶段训练方法
def train_mobilenetv4_yolo():
    # 载入模型配置
    model = YOLO('pt\yolov8s.pt')
    
    # 第一阶段：只训练骨干网络，冻结其他部分
    # 加载自定义配置文件
    custom_model = YOLO('ultralytics/cfg/models/v8/yolov8-mobilenetv4.yaml')
    
    # 从原始模型复制适合的权重
    custom_model.model = model.model  # 尝试复制权重
    
    # 冻结除骨干网络外的所有层
    for name, param in custom_model.model.named_parameters():
        if 'backbone' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    # 第一阶段训练 - 骨干网络适应
    print("Stage 1: Training backbone...")
    custom_model.train(
        data='coco.yaml',      # COCO数据集
        epochs=10,             # 第一阶段少量epoch
        imgsz=640,             # 图像大小
        batch=16,              # 批量大小
        name='stage1_backbone' # 运行名称
    )
    
    # 第二阶段：训练整个网络
    # 解冻所有层
    for param in custom_model.model.parameters():
        param.requires_grad = True
    
    # 第二阶段训练 - 整个网络微调
    print("Stage 2: Training full network...")
    custom_model.train(
        data='coco.yaml',       # COCO数据集
        epochs=100,             # 更多epoch进行完整训练
        imgsz=640,              # 图像大小
        batch=16,               # 批量大小
        name='stage2_full',     # 运行名称
        resume='runs/detect/stage1_backbone/weights/last.pt'  # 从第一阶段继续
    )

if __name__ == "__main__":
    train_mobilenetv4_yolo()