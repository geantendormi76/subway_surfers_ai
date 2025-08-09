# Subway Surfers AI - 基于模仿学习的非嵌入式游戏AI

这是一个旨在学习和复现业界前沿非嵌入式游戏AI解决方案的实践项目。项目以《地铁跑酷》为实验平台，严格对标 [wty-yy/katacr](https://github.com/wty-yy/katacr) 开源项目及其相关论文《利用强化学习玩非嵌入式卡牌游戏》，目标是完整地走通从**视觉感知**到**模仿学习决策**的端到端AI链路。

## 项目核心思想

本项目采用“模仿学习 + 迭代验证”的核心策略，分为两大阶段：

1.  **视觉感知阶段：** 采用“混合创世”策略，结合少量高质量的人工标注数据和大量程序化生成的合成数据，训练一个高性能的YOLOv8视觉感知模型。该模型负责实时、准确地识别游戏画面中的所有关键元素（玩家、障碍物、金币、道具等）。

2.  **决策模型阶段：** 采用离线模仿学习（Offline Imitation Learning）范式。首先，通过一个稳定、低延迟的PC-手机交互链路，采集人类专家的游戏录像和对应的操作序列。然后，将这些原始数据处理成结构化的 **(State, Action, Reward, Return-to-Go)** 轨迹数据集。最后，使用这些数据集训练一个基于Transformer架构的决策模型（如StARformer），使其学会模仿专家的游戏策略。

## 技术栈

*   **AI核心：** Python, PyTorch, YOLOv8, ONNX Runtime
*   **PC-手机交互：** Windows 11, WSL2, Scrcpy, ADB, uiautomator2
*   **数据处理：** OpenCV, Pillow, Numpy, Scipy
*   **配置管理：** YAML

## 项目结构

```
subway_surfers_ai/
├── data/                  # 存放所有数据，包括原始视频、轨迹、模型训练数据等 (不上传)
├── models/                # 存放训练好的模型文件 (不上传)
├── subway_surfers_ai/     # 核心Python包
│   ├── decision/          # 决策模型相关（模型定义、配置、数据加载器）
│   ├── perception/        # 感知相关（状态构建器）
│   └── utils/             # 通用工具函数
├── scripts/               # 数据处理和辅助脚本
├── train_yolo_model.py    # (原train.py) YOLO感知模型训练脚本
├── train_decision_model.py# 决策模型训练脚本
├── play.py                # AI实时游戏验证脚本
├── yolo_model_config.yaml # YOLO训练配置文件
└── decision_model_config.yaml # 决策模型训练配置文件
```

## 安装指南

1.  **克隆仓库:**
    ```bash
    git clone [您的仓库地址]
    cd subway_surfers_ai
    ```

2.  **创建并激活Conda环境:**
    ```bash
    conda create -n subway_ai python=3.9
    conda activate subway_ai
    ```

3.  **安装核心依赖:**
    请根据您的CUDA版本，从PyTorch官网安装对应的PyTorch版本。然后运行：
    ```bash
    pip install -r requirements.txt
    ```

4.  **环境配置:**
    *   **Android SDK:** 确保已安装Android Studio，并将其 `platform-tools` 路径添加到系统环境变量中。
    *   **Scrcpy:** 确保已安装 [Scrcpy](https://github.com/Genymobile/scrcpy)。
    *   **手机端初始化 (仅需一次):**
        1.  使用**USB数据线**连接手机与电脑。
        2.  开启手机的“USB调试”和“无线调试”。
        3.  在终端运行 `python -m uiautomator2 init`，允许在手机上安装ATX应用。

## 使用说明

### 1. 训练感知模型 (YOLO)

*   **准备数据：** 按照`scripts`目录下的脚本顺序（01-06），处理您的原始视频和标注，生成YOLO格式的数据集。
*   **配置：** 修改 `yolo_model_config.yaml` 文件，指定数据路径和训练参数。
*   **训练：**
    ```bash
    python train_yolo_model.py --config yolo_model_config.yaml
    ```

### 2. 采集专家数据

*   **连接手机：** 确保手机和PC在同一WiFi下，并开启无线调试。
*   **运行脚本：**
    ```bash
    # (此脚本在scripts/07_...py中)
    python scripts/07_dongzuo_jilu.py
    ```
    这将同时录制视频和动作日志。

### 3. 生成轨迹数据

*   **定位锚点：** 使用视频播放器，找到第一个动作在视频中生效的**精确帧号**。
*   **生成数据：**
    ```bash
    python scripts/08_shengcheng_guiji_shuju.py \
      --video data/raw_videos/gameplay_01.mp4 \
      --actions data/raw_videos/gameplay_01_actions.txt \
      --output data/trajectories/gameplay_01.pkl.xz \
      --first-action-frame [您找到的帧号]
    ```

### 4. 训练决策模型

*   **配置：** 修改 `decision_model_config.yaml`，指定轨迹数据目录和训练参数。
*   **训练：**
    ```bash
    python train_decision_model.py --config decision_model_config.yaml
    ```

### 5. 运行AI进行游戏

*   **准备环境：**
    1.  通过 `adb connect [IP]:[PORT]` 连接手机。
    2.  在一个终端运行 `scrcpy -e` 打开手机屏幕镜像。
    3.  手动在手机上开始一局游戏，直到角色处于待命状态。
*   **启动AI：**
    ```bash
    python play.py
    ```
