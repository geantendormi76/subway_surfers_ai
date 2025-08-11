# 《Subway Surfers AI》—— 探索纯视觉自主智能的边界

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()

**《Subway Surfers AI》是一个基于纯视觉输入的端到端非嵌入式游戏AI项目。它不读取任何游戏内存，仅通过分析屏幕画面，像人类玩家一样做出决策，并控制游戏角色。**

本项目不仅是一个游戏辅助工具，更是一个严肃的工程实践平台。我们旨在系统性地学习、复现并探索为代表的业界前沿AI解决方案，并致力于将这些技术应用于更广泛的领域，如**提升游戏可及性（为操作不便的玩家提供辅助）、工业自动化检测、以及更通用的汽车纯视觉智能驾驶研究**。

---

## 核心理念与愿景

> 我相信，游戏的乐趣在于策略与思考，而非操作的快慢。对于像我朋友一样手残脑残、手脑极度不协调的玩家，AI不应是破坏公平的“外挂”，而应是弥合生理差距、让我们能平等享受游戏核心乐趣的“智能义肢”。—— 项目发起人

本项目的核心愿景是探索AI作为**辅助技术（Assistive Technology）**的潜力，让每个人都能跨越生理的障碍，公平地参与到数字世界的互动与创造中。

我们的工程哲学遵循三大基石原则：
1.  **数据驱动 (Data-Driven):** 坚信高质量、大规模、多样化的数据是AI能力的唯一源泉。
2.  **迭代验证 (Iterative Validation):** 采用“假设-实施-诊断-验证”的科学闭环，小步快跑，持续演进。
3.  **配置与代码分离 (Configuration & Code Decoupling):** 保证实验的可复现性与工程的健壮性。

## 技术架构总览

项目将复杂的端到端AI任务，清晰地解耦为两大独立但相互协作的模块：

1.  **感知模块 (Perception):** AI的“**眼睛**”。基于**YOLO**，负责将原始的屏幕像素转化为结构化的游戏状态信息。
2.  **决策模块 (Decision):** AI的“**大脑**”。基于**StARformer (Transformer)** 架构，根据当前状态及历史信息，推理出最佳动作。


---

## 核心工作流详解

### 1. 感知引擎的锻造：YOLO数据飞轮

我们采用“**混合创世 (Hybrid Genesis)**”策略，以最小的人工成本，通过程序化生成海量数据，训练出强大的YOLO感知模型。

1.  **冷启动 (Cold Start):** 仅需手动标注约300张“困难样本”。
2.  **数字资产库构建:** 利用SAM模型，将标注物体精确抠图，创建包含所有游戏元素的“高清数字资产库”。
3.  **数据放大 (Data Amplification):** 程序化地将“标准件”随机组合粘贴到真实背景上，创造出数千张高质量的合成训练图像。
4.  **模型迭代:** 将真实数据与合成数据合并，训练出泛化能力极强的感知模型。

### 2. 决策核心的基石：工业级数据流水线

这是本项目的核心创新与关键难点。我们构建了一套自动化、时序精准的数据处理流水线，从根本上解决了模仿学习中的**因果错位 (Causal Misalignment)** 问题。

1.  **数据源革命:** 放弃个人录制，拥抱互联网上的海量高水平玩家视频。
2.  **时间基准统一:** 使用`ffmpeg`将所有视频强制转换为**恒定帧率 (CFR)**，为精准时序分析提供物理保障。
3.  **模型辅助标注:**
    *   训练一个初步的**动作识别种子模型** (`action_recognizer_best.pt`)。
    *   使用种子模型为海量视频自动生成“**草稿动作日志**”。
4.  **精准时序对齐:**
    *   提取玩家运动的“**视觉指纹**”（如Y坐标序列）。
    *   使用**动态时间规整 (DTW)** 算法，将视觉指纹与草稿动作日志进行非线性对齐，实现帧级别的精准同步。
5.  **最终轨迹生成:** 合并状态与精准动作，通过**奖励塑形 (Reward Shaping)** 计算稠密的即时奖励`R`和未来期望总回报`RTG`，生成最终用于训练的 **`(S, A, R, T, RTG)`** 轨迹。

### 3. 决策大脑的训练：从模仿到优化

我们采用“**模仿学习预训练 + 强化学习微调**”的两阶段范式。

#### **阶段一：模仿学习预训练**

*   **模型:** `StARformer`，一个基于Transformer的强大序列模型。
*   **训练:** 在高质量轨迹数据上，训练模型学习一种**条件化策略** `P(Action | State, RTG)`。
*   **过拟合对抗体系:**
    *   **科学验证与早停:** 监控验证集损失，在模型泛化能力达到顶峰时自动停止训练。
    *   **损失函数重构:** 采用类别加权和标签平滑，迫使模型学习关键动作。
    *   **高级正则化:** 通过学习率调度和降低模型容量，提升模型的抽象和泛化能力。

#### **阶段二：RLVR微调 (未来工作)**

*   **目标:** 突破模仿学习的数据上限，让模型学会“创造”最优策略。
*   **核心思想 (源自《RLVR-World》):** 将训练目标从“模仿得像”转变为“做得好”，通过最大化一个与游戏目标（如生存时长、得分）相关的**可验证奖励 (Verifiable Reward)** 来优化模型。

## 如何开始

### 1. 环境配置

本项目在**Windows + WSL2 (Ubuntu)** 环境下开发。核心训练任务在WSL2的CUDA环境中执行。

1.  克隆本仓库：
    ```bash
    git clone https://github.com/geantendormi76/subway_surfers_ai.git
    cd subway_surfers_ai
    ```
2.  创建并激活Conda环境：
    ```bash
    conda create -n subway-ai python=3.10
    conda activate subway-ai
    ```
3.  安装所有依赖：
    ```bash
    pip install -r requirements.txt
    ```

### 2. 运行AI

1.  **连接手机:** 确保您的安卓手机已开启“无线调试”功能，并与电脑处于同一局域网下。
2.  **配置IP:** 修改`play.py`文件中的`DEVICE_IP`为您手机的IP地址和端口。
3.  **启动游戏:** 在手机上打开《地铁跑酷》并开始一局游戏。
4.  **运行AI:**
    ```bash
    python play.py
    ```

### 3. 训练自己的模型

1.  **准备数据:** 按照“核心工作流”中的描述，准备视频数据并运行数据处理流水线。
2.  **训练感知模型:**
    ```bash
    python train_yolo_model.py --config yolo_model_config.yaml
    ```
3.  **训练决策模型:**
    ```bash
    python train_decision_model.py --config decision_model_config.yaml
    ```

## 贡献

我们欢迎任何形式的贡献！如果您有任何想法、建议或发现了Bug，请随时提交 [Issues](https://github.com/geantendormi76/subway_surfers_ai/issues) 或 [Pull Requests](https://github.com/geantendormi76/subway_surfers_ai/pulls)。

## 致谢

本项目的架构和核心思想深受以下优秀开源项目和研究的启发，在此表示诚挚的感谢：
*   [wty-yy/katacr](https://github.com/wty-yy/katacr): 非嵌入式游戏AI的黄金标准实现。
*   [RLVR-World: Training World Models with Reinforcement Learning](https://arxiv.org/abs/2505.13934): 提供了强化学习微调的先进思想。

## License

本项目采用 [Apache-2.0 license](LICENSE)。