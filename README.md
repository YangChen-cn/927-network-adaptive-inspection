# 927 — Network-Adaptive Visual Inspection for Industrial IoT

PCB 缺陷视觉检测原型。本仓库当前处于**第一阶段**：在 Mac 上把「数据集 → 微调 YOLO → 检测出结果」这条视觉链路完整跑通，并把后续自适应实验所需的**分辨率旋钮**提前量化出来。

> 第二阶段（device/edge 双端拆分、网络模拟、自适应放置策略、Dashboard、多策略对比）尚未实现，但代码结构已为其预留位置。

📌 **当前进度与恢复步骤见 [`HANDOVER.md`](HANDOVER.md)**
📊 **训练与实验数据汇总见 [`RESULTS.md`](RESULTS.md)** —— mAP50 = 0.9266，含逐类指标与「精度 vs 分辨率」核心实验。

---

## 1. 快速开始

### 环境

已在 **Apple M4 / macOS 27 / Python 3.14** 上验证。依赖均有 macOS arm64 轮子，无需 conda。

```bash
# 克隆后进入项目根目录
git clone <repo-url> && cd 927-network-adaptive-inspection
python3 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/pip install -r requirements.txt
```

验证安装：

```bash
.venv/bin/python -c "import torch, ultralytics; print(torch.__version__, torch.backends.mps.is_available())"
# 期望输出形如: 2.14.0 True   （True = 可用 Apple GPU 加速）
```

> 若 `pip install` 失败，回退到生态最稳的 Python 3.12：
> ```bash
> conda create -n pcbvis python=3.12 -y && conda activate pcbvis
> pip install -r requirements.txt
> ```

### 跑通全流程

```bash
# 1) 下载数据集并转成 YOLO 格式（约 1 GB，首次几分钟）
.venv/bin/python -m pcbvis prepare

# 2) 微调 YOLO（M4 上约 30~60 分钟）
.venv/bin/python -m pcbvis train

# 3) 评估精度
.venv/bin/python -m pcbvis eval

# 4) 推理并输出带框图
.venv/bin/python -m pcbvis predict --imgsz 640
```

所有命令通过 `python -m pcbvis <命令>` 调用，`python -m pcbvis -h` 可看完整参数。

---

## 2. 数据集

**`RobotHuman/PCB_defect`** — 即经典的 **PKU-Market-PCB**（Huang et al., 2019, *A PCB Dataset for Defects Detection and Classification*）。

| 项 | 值 |
|---|---|
| 规模 | 693 张图，每图 3~5 个缺陷框 |
| 类别 | `missing_hole` `mouse_bite` `open_circuit` `short` `spur` `spurious_copper` |
| 分辨率 | 3034 × 1586 |
| 标注 | Pascal VOC XML（自动转 YOLO） |
| 许可 | MIT |
| 下载 | HuggingFace 公开，**无需登录或 API key** |

### ⚠️ 使用时必须知道的三件事

1. **缺陷是合成的。** 数据集作者用 Photoshop 把缺陷合成到真实 PCB 照片上（数据集自述），**不是产线真实缺陷**。论文里必须写明这一点，它影响结论的泛化性。
2. **该镜像只有 693 张**，约为原数据集（1386 张）的一半。
3. **缺陷框极小** —— 中位宽度仅约占图宽的 2~3%（3034px 图上约 60~70px）。这是本项目最关键的物理事实：**推理分辨率一降，小缺陷最先检不出来**，这正是「分辨率 / 时延 / 带宽」权衡的真实来源，而非人为构造。

### 标注转换的两个坑

代码里已处理，此处记录以免日后踩坑：

- XML 中的 `<filename>` 字段（`01_missing_hole_01.jpg`）**与实际文件名（`missing_hole01.jpg`）不一致**。配对只能按文件名词干做，不能信 `<filename>`。
- 类别名以 XML 内 `<name>` 为准，不要从文件夹名推断。
- 图像尺寸以**图片实际尺寸**为准（`PIL` 读取），XML 的 `<size>` 仅用于交叉校验；不一致会打印警告。

---

## 3. 目录结构

```
.
├── configs/pcb.yaml        # 类别表、划分比例、随机种子
├── pcbvis/                 # 主包
│   ├── paths.py            # 统一路径管理
│   ├── config.py           # 配置读取
│   ├── data_prep.py        # 下载 + VOC→YOLO 转换 + 分层划分
│   ├── train.py            # 微调（自动选 MPS）
│   ├── predict.py          # 推理 + 画框 + 逐图时延
│   ├── evaluate.py         # mAP 指标
│   └── cli.py              # 命令行入口
├── data/
│   ├── raw/pcb_defect/     # 原始下载（自动生成，不入库）
│   └── pcb_yolo/           # 转换后数据集 + data.yaml
├── models/                 # 预训练 + 微调权重
└── results/
    ├── train/              # 训练输出（best.pt / 曲线图）
    ├── eval/               # metrics_imgsz{N}_{split}.json
    └── predict/imgsz{N}/   # 带框图 + predictions.json
```

---

## 4. 分辨率旋钮（本项目的核心）

`--imgsz` 贯穿推理与评估，直接对应第二阶段的权衡变量：

```bash
# 精度 vs 分辨率曲线（一次跑完，直接出表）
.venv/bin/python -m pcbvis sweep --imgsz 320,416,640,1280

# 看不同分辨率下的实际检测效果（带框图会叠加 imgsz 与时延）
.venv/bin/python -m pcbvis predict --imgsz 320    # 快，但小缺陷大量漏检
.venv/bin/python -m pcbvis predict --imgsz 1280   # 慢，召回更全
```

每次 `predict` 都会把**逐图推理时延**写进 `predictions.json`，这是第二阶段估计「本地推理耗时」的实测依据；`eval` 的逐 imgsz 结果则是「精度-分辨率」查找表的标定数据。

---

## 5. 常用参数

| 命令 | 参数 | 说明 |
|---|---|---|
| `prepare` | `--no-download` | 跳过下载，只重做转换 |
| | `--limit N` | 每类只取 N 张，冒烟测试用 |
| `train` | `--epochs` `--batch` `--imgsz` | 训练超参 |
| | `--device mps\|cpu` | 默认自动探测；MPS 异常时用 `cpu` |
| | `--workers N` | macOS 上 DataLoader 卡住时设 `0` |
| `predict` | `--imgsz` `--conf` `--limit` | 推理分辨率 / 置信度阈值 |
| `eval` | `--split val\|test` | 评估子集 |

---

## 6. 复现性

- 划分种子固定为 `42`（`configs/pcb.yaml`），按类别分层，比例 70/15/15。
- 训练超参写入 `results/train/pcb_yolo11n/run_config.json`。
- 训练曲线图由 ultralytics 输出在 `results/train/pcb_yolo11n/`。

---

## 7. 第二阶段路线（尚未实现）

1. **网络模拟器** —— 可配置带宽 / RTT / 抖动 / 丢包（用户态模拟，字节数真实、时延按模型计算）。
2. **edge server** —— 独立进程提供 HTTP 推理服务，可模拟服务器负载。
3. **自适应策略** —— 基于实测时延与「精度-分辨率」标定表，在线选择「本地 / 卸载」与分辨率；目标是最小化时延与传输量，同时满足精度下限与截止时间。
4. **对比实验** —— local-only / edge-only / 固定卸载 / 自适应，在不同网络条件下比较精度、端到端时延、截止时间达成率、传输数据量。
5. **演示界面** —— 实时显示检测框与系统决策。
