# DeepLearningLab：从 Python 初学者到独立实现现代深度学习算法

> 版本：v1.0（2026-10-10）  
> 文件作用：**长期课程总纲 + 课程验收标准 + 跨 Agent 交接依据**。  
> 配套文件：`AGENTS.md`（Codex 的常驻约束）、`PROGRESS.md`（唯一可信的实际进度）、`.agents/skills/deep-learning-mentor/SKILL.md`（教学工作流）。  
> 项目根目录：`~/projects/DeepLearningLab/`。

## 0. 读者说明：请新 Agent 先看这里

本项目是一门以**学生亲手写代码**为核心的长期、个性化编程课程。学生是计算机与机器学习方向研究生，有一定 Python 基础（知道变量赋值、函数、类的概念），但缺乏 Python 进阶知识、依赖/环境管理经验、PyTorch API 和从零搭建工程的经验。以前经常让 AI 直接生成代码，因此本课程必须反向纠正这一习惯。

**最终学习目标不是“跑通一个教程”，而是在不用 AI 代写核心算法的条件下：**

1. 自己设计、编写、调试并维护 Python/PyTorch 研究代码。
2. 看懂陌生深度学习项目的模块边界、数据流、Tensor Shape、数学公式、训练/采样过程。
3. 从空仓库实现并训练可运行的 mini-GPT / nanoGPT 风格语言模型。
4. 从零实现 Transformer、VAE、GAN、Normalizing Flow、Diffusion（DDPM/DDIM/score 视角）、Flow Matching / Rectified Flow 等核心算法。
5. 迁移这些能力到视觉、三维、动态人体、3D Gaussian Splatting、生成式建模等研究代码（选修专题，不抢占基础课程）。
6. 能独立定位报错、复现实验、做消融、为代码编写测试并评估是否符合论文原理。

**特别注意：**截至本计划建立时，学生仅在先前对话中**听过**第 1 节课的说明，尚未提交作业，也未验证其掌握情况。不能把“讲过”当作“已通过验收”。具体状态以 `PROGRESS.md` 为准。

## 1. 真实环境和现有项目状态（2026-10-10）

### 1.1 用户实际环境（学生自述）

- 系统：Ubuntu / Linux；本地有 NVIDIA GPU（此前交流中使用过约 96 GB 显存的 RTX PRO 6000 级别设备；**Agent 不得据此假定本次机器状态，使用前实测**）。
- CUDA 方向：12.8。
- Conda 环境：**`dllab`**（不是最早教学示例中的 `learn-gpt`）。
- Python：**3.11**。
- PyTorch：**2.10.0**。
- 使用本地 Codex 进行大部分课程，也会返回网页版 ChatGPT 讨论。
- 已有目录：

```text
~/projects/DeepLearningLab/
├── lessons/
├── nanoGPT/
└── README.md
```

- `nanoGPT/` 的现有内容、Git 状态、环境实际可用性都**尚未由导师验证**；开始前只读检查，**不得覆盖或重置**。

### 1.2 环境操作约定

- 首次仅检查（无需重新创建）：`conda info --envs`、`python --version`、`python -m pip show torch`。
- 按需检查 CUDA：`python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'`。
- 确认 `which python` / `sys.executable` 指向 `dllab`，理解“驱动 CUDA / Toolkit / PyTorch CUDA wheel”的区别。
- **除非学生明确同意，否则不要安装/升级/降级包、创建第二个 Conda 环境、清理文件或覆盖已有模型权重。**
- 环境不兼容时，先给出证据、原因、最小修复方案，再让学生执行；不要无缘无故修改 CUDA、Torch 版本。
- GPU 大显存只影响可运行的实验规模，不应跳过基础训练。默认使用 CPU 或很小的 GPU 张量学习，逐步扩大。

### 1.3 推荐逐步扩展的目录（不是立即要求创建）

```text
DeepLearningLab/
├── AGENTS.md                         # Codex 常驻规则（本包提供）
├── LEARNING_PLAN.md                  # 本文：课程路线（本包提供）
├── PROGRESS.md                       # 学习记录（本包提供）
├── .agents/skills/deep-learning-mentor/SKILL.md
├── README.md                         # 学生已有；不覆盖
├── lessons/
│   ├── 01_tensor.py                  # 第一份拟提交作业（不假定存在）
│   ├── ...
│   └── notes/                        # 可选：学生自己的笔记
├── nanoGPT/                          # 学生已有；先检查后再决定如何使用
├── projects/                          # 后续：从头实现的算法项目
│   ├── mlp_from_scratch/
│   ├── vision_classifier/
│   ├── mini_gpt/
│   ├── vae/
│   ├── ddpm/
│   └── flow_matching/
├── paper_reading/                    # 后续：论文公式到代码的映射
└── tests/                            # 后续：算法正确性测试
```

**不要一次创建所有目录**；课程进展到对应专题时，学生应亲自解释为何引入目录并创建它。`nanoGPT/` 和未来的 `projects/mini_gpt/` 可分别用作参考阅读和本人手写实现（若已有内容不符，先协商命名）。

## 2. 核心教学契约（所有 Agent 必须遵守）

### 2.1 教学原则

1. **学生是代码作者，Agent 是导师、审阅者和调试教练。**默认不直接提供完整、可复制粘贴的一整个实现文件或完整实验项目。
2. 按学生 2026-10-13 明确的新偏好授课：**本章节内容与意义 → 原理/数学/运行机制 → Python/PyTorch API 的用法与适用场景 → 任务意义与目标 → 学生手写 → 集中反馈**。开章先说明解决什么实际编程问题、为什么需要学及如何联系后续算法。
3. 默认不提供可直接运行的代码，包括局部完整示例；使用文字、公式、Shape 表、接口签名及不可直接执行的伪代码。说明 API 参数、返回值、副作用及用法，由学生自己实现。仅在学生明确另行要求参考实现时调整。

   2026-10-13 学生补充要求：适当举例并详细解释抽象概念，不能只列术语或一句定义。采用“明确含义 → 具体数值/公式/图示 → 对照判断 → 实际用途与常见误解”的方式；如 is_leaf 与 grad_fn，应展示直接创建的输入和被追踪运算结果的区别，以及它们与梯度保存的联系。学生反馈未理解时先补讲，不额外增加验收门槛；示例不默认包含可运行代码。
4. **每次给出一个范围适当的完整任务**。按学生 2026-10-12 的新偏好，把相关概念合并为完整课块讲解、集中练习与验收，减少每个 API 单独往返；遇到明确障碍再拆分，不一次堆积无关知识。
5. 错误反馈先指出位置、现象、假设和检查手段；必要时逐级加提示，不上来粘贴修改后的完整代码。
6. 每个 Python 关键语法第一次出现，解释其含义与常见用法：例如 `*args`、`**kwargs`、继承、`super()`、`@property`、`@staticmethod`、`@torch.no_grad()`、`__call__`、上下文管理器、类型注解等。
7. 对每一个核心 Tensor 运算回答：**输入 Shape / 输出 Shape / dtype / device / 是否参与计算图**；出现 `reshape`、`transpose`、`permute`、`matmul`、广播时尽量用小尺寸算例说明。
8. 对模型与算法回答：**问题定义 → 目标函数 → 参数与数据 → 前向传播 → 损失计算 → 反向传播 → 推理或采样 → 验证方式**。
9. 明确区分“学生能复述”“学生能修改”“学生能独立写出并解释”；只有通过验收才标记 mastered。
10. 尊重学生的真实代码，避免未经允许的大规模重构、文件覆盖、偷偷补全答案。
11. 语气友好、细致、不居高临下；不因卡在基础知识而跳过讲解。
12. 追求可迁移能力：同一概念分别在语言、图像和生成模型里认出来。

### 2.2 推荐的 Socratic 提示阶梯

- L0：复述要求和输入/输出，不给实现。
- L1：指明应查阅的 API/语法、数学关系或相关文件。
- L2：不可直接执行的伪代码/接口签名、Shape 表、文字描述的局部操作。
- L3：指出具体 Bug 及局部修改策略。
- L4：仅在用户明确另行要求代码参考时提供局部参考，并解释为什么；**不要伪装成学生独立完成**。多次受阻时先补充原理与用法，不自动突破默认“不提供可直接运行代码”的偏好。

学生随时可选择“只要提示 / 详细讲解 / 代码审阅 / 参考答案”，默认 L1–L2；遇到硬性阻碍（驱动/安装失败）可用完整**命令**协助，但核心训练/算法代码仍保持教学优先。

### 2.3 每节课的固定流程

1. **开课前**：每次开始或恢复课程，重新读取磁盘上的 `AGENTS.md`、`PROGRESS.md` 当前状态与最近实际 Session Log，以及本计划对应模块；确认章节、验收状态和最小任务，只读观察所需代码。切换设备或 Git pull 后同样执行，不凭聊天记忆跳课。
2. **开章介绍与必要复习**：先说明本章内容、学习意义、实际用途及与前后章节的联系；有明确前置缺口时直接提问，不机械重复已通过内容。
3. **讲解与用法**：解释概念、数学、Shape 与机制；逐项说明 API 参数、返回值、副作用、适用场景及操作方法，不给可直接运行的代码。
4. **布置任务**：首先说明任务的意义、能力目标和完成后能解决什么问题，再明确文件路径、输入输出、核心验收条件及参考 API；不默认要求把预测或答案写成注释。
5. **学生编写**：鼓励学生先自己写并运行，不替代键盘。
6. **代码 Review**：先验证实际行为，再逐项解释问题；引导学生自行修复。
7. **验收**：代码在当前教学环境运行正常、结果正确并体现本章核心内容即可通过。核心算法/数学/数据流错误需修正；非核心兼容性、打印/命名/格式等作为改进建议，不反复阻塞进度。有明确理解疑点时直接提出针对性问题，学生自由选择聊天或注释回答；不强制每课边界测试、变体或整张量预测。
8. **记录**：学生已于 2026-10-12 持续授权自动记录。每次章节/小节/作业验收后立即更新 `PROGRESS.md`，无需再次询问是否记录；含日期、文件、实际证据、结论、提示依赖与下一步。只有真实通过才标记 `PASSED`；未通过记录剩余任务。路线或规则改变时同步维护对应指导文档。建议将作业与记录一起提交 Git。

### 2.4 验收维度（统一评分建议）

每次作业从 0–2 分评估：

| 维度 | 0 | 1 | 2 |
|---|---|---|---|
| 语法与 API | 不理解 | 借助提示完成 | 独立使用并解释 |
| 数学与 Shape | 不清楚 | 部分说清 | 能手算/追踪变形 |
| 正确性 | 失败 | 主路径成功 | 边界与小测试成功 |
| 调试能力 | 依赖 Agent 改 | 会定位 | 能建立假设并验证 |
| 工程质量 | 混乱 | 可运行 | 可读、模块化、可复现 |

**用途与通过标准（2026-10-13 更新）**：本表用于诊断与安排复习，不作为逐项满分的硬性通过门槛。基础章节以代码可运行、结果正确且体现核心知识为通过标准；无提示独立熟练度、工程细节及非核心边界完善程度分别记录。项目阶段以明确的核心功能和算法正确性验收，不以无关细节阻塞。真实掌握证据仍须保留，不能把运行成功等同于全面熟练。

## 3. 课程路线图与依赖关系

### 当前采用的授课方式（2026-10-12 学生确认）

以下路线描述知识依赖，实际教学以 **C 阶段 PyTorch 及后续深度学习实践为主线，A/B 按需穿插并与实践联系**，不机械要求先完整重学 A/B 再进入 C。

学生自述已了解 A/B 中许多概念，而深度学习编程与 PyTorch 实践经验相对不足，因此优先用实践组织课程。此项仅记录学习背景与偏好，不将“了解概念”认定为独立实现能力或完整验收通过。

- 熟悉的 Python/数学基础通过简短检查、小变体或实践解释确认，避免重复完整授课。
- 新 API 与算法显式联系其 Python 语法、数学原理、Shape 和工程用途；必要前置概念应在使用前讲清。
- 发现基础缺口时，插入范围明确的 A/B 课块再返回主线，不等到报错后才零碎补一句。
- A/B 课程内容与 P-A/P-B 验收项目保留；后续结合工程或算法实践安排验收，不因 C 阶段推进而自动升级 A/B 状态。
- 相关概念合并为完整课块，集中练习和运行验收；先定义术语与 API，再讲机制，避免只增加概念密度。

近期衔接示例：C03 联系 B03/B04（广播、向量化、点积与矩阵乘法）；C05 联系 B05（偏导数与链式法则）；C07 联系 A03/A05/A06（函数、类、继承与 super）；多文件训练工程联系 A07/A08/A12（包、路径、配置与入口）。具体进度以 PROGRESS.md 为准。

主线建议按顺序推进：

`Python → NumPy/数学 → PyTorch Tensor/Autograd → MLP/CNN/训练 → 工程化 → Transformer/mini-GPT → VAE/GAN/Normalizing Flow → Diffusion → Flow Matching → 扩展算法 → 阅读真实项目与研究复现`

**可交叉训练**：从阶段 2 开始持续做源码阅读；从阶段 3 开始持续写测试；工程与调试贯穿始终。不要为了追赶日期跳过关键前置条件。

下列建议课次编号是**课程索引**，不是已存在的文件，也不是固定学时。每课视难度可拆成几次对话。

## 4. 阶段 A：Python 进阶与工程语法（A01–A12）

**目标**：不再把 Python 代码视为模板，而是理解语法、运行机制与可维护的组织方式。

| 课次 | 学习主题 | 学生亲手任务 / 迁移场景 |
|---|---|---|
| A01 | 基础回顾：对象、变量绑定、可变与不可变、`id`/浅拷贝 | 预测修改列表/张量变量后的效果 |
| A02 | 列表、元组、字典、集合；推导式、切片与解包 | 从嵌套训练配置中提取/整理参数 |
| A03 | 函数：默认参数、关键字参数、作用域、返回多个值 | 写参数合法性检查函数 |
| A04 | `*args`、`**kwargs`、匿名函数、闭包 | 理解模型初始化和配置传递 |
| A05 | 类与对象：`__init__`、实例/类属性、方法、`self` | 写简易 `MetricTracker` |
| A06 | 继承、`super()`、特殊方法、`@property` | 建立类层次；为 `nn.Module` 做准备 |
| A07 | 模块与包：`import`、相对/绝对导入、`__init__.py`、`__name__` | 把脚本改为可导入的多文件项目 |
| A08 | pathlib、文件读写、JSON/YAML 与路径 | 读取配置、处理日志文件 |
| A09 | 异常、断言、日志、调试、栈追踪 | 自己定位 3 种常见报错 |
| A10 | 迭代器/生成器、`yield`、`enumerate`、`zip` | 实现简单 batch 迭代器 |
| A11 | 上下文管理器、装饰器、类型标注、dataclass | 理解 `no_grad` 和配置类 |
| A12 | `argparse`、入口文件、环境/依赖、Git 基础、测试简介 | 可执行的多模块 CLI 小工程 |

**验收项目 P-A**：自建 `lessons/python_toolbox/`（名称可自定），实现小型数据读取 + 配置 + 统计 CLI。能解释 Python 包是如何被 import 的，`python file.py` 与 `python -m package.module` 有何不同。

**常见易错点**：默认参数放可变对象、`is` vs `==`、浅拷贝、循环变量作用域、循环 import、工作目录与路径、包的搜索路径。

## 5. 阶段 B：数学、NumPy 与张量思维（B01–B09）

**目标**：数学公式能落地成 Tensor Shape / 运算步骤，并具备数值直觉。

| 课次 | 学习主题 | 学生亲手任务 |
|---|---|---|
| B01 | 标量/向量/矩阵/张量、维度、轴 | 手写多个 Shape 与索引结果 |
| B02 | NumPy ndarray、切片、view/copy、连续存储 | 验证内存共享与修改影响 |
| B03 | 广播规则、批处理、向量化 | 用循环与广播实现同一算式 |
| B04 | 点积、矩阵乘法、转置、矩阵微积分 | 手算小矩阵与梯度 |
| B05 | 链式法则、梯度、Jacobian、偏导数 | 手推线性回归导数并做数值差分 |
| B06 | 概率、期望、方差、Gaussian、采样 | 模拟高斯分布的统计量 |
| B07 | 对数概率、最大似然、交叉熵、KL | 比较 NLL / CE / KL 含义 |
| B08 | 优化：SGD、Momentum、Adam、学习率 | 手算 1–2 步参数更新 |
| B09 | 数值稳定性：log-sum-exp、softmax、精度 | 对比朴素/稳定的 softmax |

**验收项目 P-B**：纯 NumPy 实现线性回归拟合（包括 loss 和手动梯度），与解析解或数值梯度核对；解释 batch 与维度。

## 6. 阶段 C：PyTorch 核心语言（C01–C11）

**目标**：能独立完成张量表达、设备管理、Autograd 和最小训练循环。

| 课次 | 学习主题 | 学生亲手任务 |
|---|---|---|
| C01 | Tensor 构造、dtype、device、shape、索引、切片 | **先完成初始作业 `lessons/01_tensor.py`** |
| C02 | `reshape`、`view`、`permute`、`transpose`、`unsqueeze`/`squeeze` | 明确每一步维度变化 |
| C03 | 广播、`@`/matmul、`einsum` 入门 | 手算小尺寸批量矩阵乘法 |
| C04 | 设备、CPU↔CUDA、非原地与原地操作、`.to()` | 检测 device mismatch 并修复 |
| C05 | 计算图、`requires_grad`、叶子 Tensor、`backward` | 对比手算和 autograd |
| C06 | 梯度累积、`detach`、`no_grad`、`inference_mode` | 展示何时没有梯度 |
| C07 | `torch.nn.Parameter`、`nn.Module`、`forward`/`__call__` | 手写一个最小线性层类 |
| C08 | 损失函数、优化器、`zero_grad/backward/step` | 完成线性回归训练循环 |
| C09 | `Dataset`、`DataLoader`、batch、shuffle、collate | 实现自定义 Dataset |
| C10 | 初始化、随机种子、保存/加载 `state_dict` | 恢复权重后复现推理 |
| C11 | GPU 监测、内存、错误调试、基准测试、梯度检查 | 从错误 stack trace 定位问题 |

**验收项目 P-C**：只使用小规模 Tensor，自己写 `Linear` → `MSE` → `SGD` 最小训练流程；梯度与 loss 正确；解释 `model.train()` 与 `model.eval()` 的含义。

**C02 组织调整（2026-10-12，学生明确请求）**：将维度操作作为一个整体，按“增删单例轴 → 重组形状 → 交换/重排轴 → stride、连续性与 view/reshape”建立统一对照。已通过部分简短串联，剩余 permute 与 view/连续性合并讲解；集中布置一个综合练习、一次运行核对多个结果。学生仍亲手编写核心代码，理解确认并入综合验收，不逐 API 重复运行。后续课程沿用相关概念成组、集中验收的方式。

**关于最初的第 1 课（精确存档）**：先前只介绍过 `import torch`、`torch.tensor`、`.shape/.dtype/.device`、`x.mean()`、`x[0,1]`、`x[:,1]`、`mean(dim=...)`、`.to('cuda')`；布置了 2×3 浮点 Tensor、属性打印、选取行列、按列均值、移到 CUDA 的任务，含三个思考题：Shape `(2,3)` 的含义、双索引的含义、`x.to('cuda')` 是否原地修改。**没有提交/验收记录，仍是待完成**。

## 7. 阶段 D：从零训练神经网络（D01–D10）

**目标**：理解“模型类、损失、反向传播、优化器、数据、日志、验证”如何协同。

| 课次 | 学习主题 | 学生亲手任务 |
|---|---|---|
| D01 | 多层感知机 MLP、线性层、非线性、参数量 | 手写多层网络并计算参数量 |
| D02 | SGD/AdamW、权重衰减、调度器 | 对比收敛曲线 |
| D03 | 分类任务、CrossEntropyLoss、logits | 训练合成二维分类数据 |
| D04 | train/val/test、泛化、过拟合、数据泄露 | 设计对照实验 |
| D05 | BatchNorm/LayerNorm/Dropout | 区分训练和推理行为 |
| D06 | 卷积、padding、stride、pooling、输出形状 | 手写一个简化 Conv2D 运算思路 |
| D07 | CNN、ResNet、跳跃连接 | 构建图像分类器并解释残差 |
| D08 | 可重复性、checkpoint、日志、指标 | 从 checkpoint 恢复训练 |
| D09 | AMP 混合精度、梯度裁剪与累积 | 比较显存、速度、数值变化 |
| D10 | 单元测试、Shape 测试、梯度测试、过拟合单 batch | 编写能捕获实际 Bug 的测试 |

**验收项目 P-D**：完整图像分类工程（小数据集，禁止以 SOTA 为目标）。必须至少包含 `model.py`、`train.py`、`evaluate.py`、配置、训练日志、checkpoint、README、至少 3 个有意义的测试。学生自己实现核心类/训练循环。

## 8. 阶段 E：Transformer、语言建模与 mini-GPT（E01–E14）

**目标**：手写最小可训练的 Decoder-only GPT，并理解与主流 Transformer 变体的关系。这是首个大型里程碑。

| 课次 | 学习主题 | 学生亲手任务 |
|---|---|---|
| E01 | 序列建模、token、词表、字符级 tokenization | 实现字符编码和解码 |
| E02 | n-gram、语言模型概率分解、next-token 目标 | 从原始字符串构造输入/标签 |
| E03 | Embedding、位置编码（learned/sinusoidal/RoPE 基础） | 解释 `(B,T,C)` |
| E04 | Attention 的 Q/K/V 与缩放点积 | 在最小矩阵中手算注意力 |
| E05 | Causal mask、softmax、数值稳定性 | 验证未来 token 不可见 |
| E06 | Single-head Self-Attention 独立实现 | 写模块与维度测试 |
| E07 | Multi-head Attention、head reshape/transpose | 验证多头拼接和投影 |
| E08 | Residual、LayerNorm、MLP、Pre-Norm | 手写 Transformer Block |
| E09 | Decoder-only GPT 堆叠、权重共享/初始化 | 实现模型并计算参数量 |
| E10 | 交叉熵、Teacher Forcing、训练小 GPT | 证明模型可过拟合少量文本 |
| E11 | 自回归采样、temperature/top-k/top-p | 实现文本生成器 |
| E12 | checkpoint、验证困惑度、实验记录 | 训练/推理分离并复现输出 |
| E13 | 阅读 nanoGPT 并建立论文/模块映射 | 输出数据流、调用图、参数表 |
| E14 | 复杂度、KV Cache、SDPA/FlashAttention、推理效率 | 与手写 Attention 做一致性对照 |

### P-E：mini-GPT 必须满足的验收要求

- 自己设计一个**与参考 nanoGPT 源码不逐行照抄**的小型项目。
- 有 Tokenizer、dataset/batch 采样、Causal Self-Attention、Block、GPT、训练、评估、生成、保存/恢复。
- 含可运行的小数据实验；`train loss` 下降，验证损失可报告；固定 seed 可复现关键输出（容忍设备非完全确定性）。
- 单元测试涵盖：tokenizer round trip、mask 的因果性、形状/梯度、采样范围、保存-恢复一致性。
- 能向导师解释：为什么输出 logits 的 Shape 是 `(B,T,V)`；`cross_entropy` 如何组织输入；`QK^T / sqrt(d_k)`；为什么未来信息泄露是错误；参数量的来源。
- 能说清参考项目中入口文件、模型文件、配置、数据预处理之间的关系。
- 若从零训练资源有限，优先用 toy 数据和小模型，而非盲目追大模型效果。

### E 阶段完成后的知识延伸

Encoder-only / Encoder-Decoder；Cross-Attention；RoPE 细节；RMSNorm / SwiGLU；GQA/MQA；LoRA/PEFT；预训练与微调；ViT/DiT。先打通原理再使用框架封装 API。

## 9. 阶段 F：生成建模统一视角（F01–F09）

**目标**：建立密度建模、潜变量、对抗学习、变分推断和概率流的共同框架，防止只会记模型名字。

| 课次 | 学习主题 | 学生亲手任务 |
|---|---|---|
| F01 | 生成建模：显式似然/隐式生成、采样与推理 | 建模任务比较 |
| F02 | Autoencoder：Encoder/Decoder/重建损失 | 在低维或简化图像数据上重建 |
| F03 | VAE：ELBO、重参数化、KL、潜变量 | 手推并实现基本 VAE |
| F04 | β-VAE、Posterior Collapse、潜变量插值 | 分析重建-正则权衡 |
| F05 | GAN：Generator/Discriminator、minimax | 合成二维数据训练 |
| F06 | GAN 稳定性：mode collapse、判别器失衡 | 诊断失败现象 |
| F07 | Normalizing Flow、变量代换、Jacobian | 用简单可逆映射算密度 |
| F08 | Coupling Layers / RealNVP、可逆性 | 训练二维密度模型 |
| F09 | 对比 VAE/GAN/Flows/AR/扩散与流匹配 | 写一份公式-代码-训练-采样对照笔记 |

**验收 P-F**：VAE + 2D Normalizing Flow 两个小实现；清楚区分 encoder 近似后验、flow 可逆映射、生成时从哪个分布采样；GAN 可作为附加小实验。

## 10. 阶段 G：Diffusion / Score-based 模型（G01–G12）

**目标**：能解释并独立实现前向加噪、噪声预测、反向采样，理解离散与连续时间方法。

| 课次 | 学习主题 | 学生亲手任务 |
|---|---|---|
| G01 | 马尔可夫链、Gaussian 噪声、时间步、SNR | 画不同 t 的噪声强度 |
| G02 | DDPM 前向过程，β/α/ᾱ 调度 | 直接采样 `x_t` 并验证方差 |
| G03 | 噪声预测 εθ、timestep embedding、MSE | 做 toy ε-prediction |
| G04 | DDPM 训练目标和推导（从 ELBO 到 simplified loss） | 对应每个公式到代码 |
| G05 | DDPM 反向步骤、系数、随机项、边界 | 小规模生成 2D 数据 |
| G06 | 用简化 U-Net / MLP 做图像或二维实验 | 实现条件时间网络 |
| G07 | DDIM 与确定性/非确定性采样 | 对比生成步数和误差 |
| G08 | Score matching 与去噪分数估计 | 解释 score 与 ε 的关系 |
| G09 | VP/VE SDE、概率流 ODE（直观为主） | 映射离散/连续时间记号 |
| G10 | Conditional Diffusion、Classifier-free Guidance | 条件丢弃与采样引导 |
| G11 | Latent Diffusion、VAE、UNet、DiT 概念 | 读图画出各模块数据流 |
| G12 | Scheduler 抽象、采样验证、速度/质量评估 | 对照实现并做消融 |

**关键最小公式（理解后必须能独立写出，而非照搬）**：

- `α_t = 1 - β_t`，`ᾱ_t = ∏_{s=1}^t α_s`。
- `x_t = sqrt(ᾱ_t) x_0 + sqrt(1 - ᾱ_t) ε`，其中 `ε ~ N(0,I)`。
- 简化噪声预测目标：`E_{x0,t,ε} ||ε - εθ(x_t,t)||²`。

**验收 P-G**：独立实现 toy DDPM 的前向扩散、训练和采样；检查 t=0 与最大 t 的边界、均值方差、模型输出 Shape、采样无梯度；画出不同时间步的加噪过程；比较不同采样步数。扩展到简化图像任务是进阶选项。

## 11. 阶段 H：Flow Matching、Rectified Flow 与连续时间生成（H01–H11）

**目标**：真正搞清楚“学向量场”和“数值求解 ODE”，理解与 Diffusion 的联系与区别。

| 课次 | 学习主题 | 学生亲手任务 |
|---|---|---|
| H01 | 常微分方程 ODE、速度场、轨迹、初值问题 | 一维 Euler 积分并检查误差 |
| H02 | Euler、Heun、RK4、步长与稳定性 | 比较解的精度和计算次数 |
| H03 | Continuity Equation / Probability Flow 直觉 | 区分点轨迹与概率密度变化 |
| H04 | 概率路径、线性插值路径、噪声→数据 | 在二维绘制样本轨迹 |
| H05 | Conditional Flow Matching：目标速度场与回归损失 | 实现二维 CFM 训练 |
| H06 | 推理：学习到的向量场 + ODE solver | 从标准 Gaussian 生成二维样本 |
| H07 | Coupling 的选择：独立/OT 配对（入门） | 比较路径弯曲、求解步数 |
| H08 | Rectified Flow：直线路径、速度预测与再流化概念 | 对比普通 CFM 设置 |
| H09 | Diffusion 与 Flow Matching 的概率路径/参数化关联 | 比较 ε/x0/v/velocity 参数化 |
| H10 | Conditional Flow / CFG / Time Conditioning | 添加类别条件与测试 |
| H11 | Flow-based Transformer / DiT 研究阅读入门 | 提取主要模块和 Loss |

**入门公式（以选定的线性条件路径为例，不代表所有 FM 方法）**：

- 取 `x_0 ~ p_0` 为噪声，`x_1 ~ p_data` 为数据；设 `x_t = (1-t)x_0 + t x_1`。
- 对应条件目标速度 `u_t = x_1 - x_0`。
- 训练 `L = E_{t,x0,x1} ||vθ(x_t,t) - (x_1-x_0)||²`。
- 生成通过 ODE `dx_t/dt = vθ(x_t,t)` 从 `t=0` 积分到 `t=1`。

强调：路径分布、耦合方式和时间参数化会影响目标与行为；不得把任意 Flow Matching、Rectified Flow、Diffusion 偷换成“完全一样”。

**验收 P-H**：用小 MLP 在二维 mixture of Gaussians / two moons 等合成数据上训练；自己写 Euler 与 Heun 采样并作分布、轨迹、NFE（函数评估次数）对比；解释为什么 `x_t` 与 target velocity 的定义必须匹配。

## 12. 阶段 I：重要研究算法扩展（I01–I12，按研究兴趣选修）

这些不是“毕业前全部必须做”，而是获得迁移能力的技术池。优先选能支撑用户研究项目的方向。

| 专题 | 核心算法 / 应会实现的最小模块 | 与基础阶段的连接 |
|---|---|---|
| I01 | Contrastive Learning：InfoNCE、SimCLR | 交叉熵、特征归一化、正负样本 |
| I02 | CLIP 风格图文对比学习 | 多模态对齐、双塔网络 |
| I03 | Vision Transformer / DiT | Attention 与 Patchify、AdaLN |
| I04 | LoRA / PEFT / Adapter | 矩阵低秩分解、冻结参数 |
| I05 | Mixture of Experts | Router、top-k、负载均衡 |
| I06 | State Space / Mamba 概念 | 递归、序列模型替代方案 |
| I07 | GNN：Message Passing | scatter / index_add / 邻接关系 |
| I08 | NeRF：位置编码与体渲染基本积分 | MLP、数值积分、相机射线 |
| I09 | 3DGS：Gaussian 参数、投影、alpha compositing | 多视角、渲染、几何约束 |
| I10 | 人体参数化 / 变形场 / Cloth dynamics | SMPL-X、pose conditioning、时空表示 |
| I11 | RL 基础：MDP、Policy Gradient、PPO 概念 | autograd、采样、优化 |
| I12 | 模型参数生成/Hypernetwork/Weight-space | MLP 权重、permutation、latent 建模 |

选修专题的验收：独立实现一个**可验证的小核**，并对一篇开源论文给出函数级阅读笔记。不强迫复刻高算力的整篇论文。

## 13. 阶段 J：真正能读懂陌生项目（纵向贯穿，最后综合验收）

**阅读一个陌生开源项目时统一流程：**

1. 确定任务定义：输入/输出/训练数据/评价指标/论文贡献。
2. 先看 `README`、项目目录、安装文件 `pyproject.toml`/`requirements.txt`，再找运行入口。
3. 从命令行入口追踪配置 → 数据读取 → dataset/dataloader → model 构造 → forward → loss → backward/optimizer → checkpoint → inference。
4. 记录重要类和函数的**调用图、传参含义、返回结构、Shape 表**。
5. 选一个核心模块，把论文公式逐行映射到代码；分辨“方法本身”和“工程/速度优化”。
6. 写一个不依赖原训练集的小单元测试，验证关键模块。
7. 修改一个小参数或功能，预测结果，再运行检查。
8. 针对不理解的 Python/PyTorch 语法开一张补修卡，补齐后返回。

### 源码阅读训练梯度

- J01：教学级 MLP/CNN 仓库（能画训练循环）。
- J02：nanoGPT（能独立复述每个关键模块）。
- J03：官方 PyTorch Transformer 示例 / 简单 ViT（处理真实模块拆分）。
- J04：小型 DDPM / Diffusers 训练示例（理解 scheduler 与 loss）。
- J05：Flow Matching 实现（理解 `sample t → construct x_t → target velocity → optimizer`）。
- J06：学生研究方向中的真实复杂项目（3DGS、人体 avatar、模型参数生成等），只选**一个**核心算子做深读。

**综合毕业任务 P-J**：任意选一篇近年的开源论文，在不让 Agent 代写算法的条件下，独立产出：（a）1 页算法摘要；（b）重要公式与文件/函数对照；（c）端到端数据流和 Shape；（d）一个最小复现模块；（e）测试与消融；（f）README 和限制说明。

## 14. 贯穿课程的工程技能清单

| 能力 | 最迟何时引入 | 具体掌握标准 |
|---|---|---|
| Conda / pip / 版本与 CUDA | 阶段 A | 能定位当前 Python 和依赖版本冲突 |
| 目录/包/导入与入口 | 阶段 A | 能组织 3+ 文件的可导入项目 |
| Git 分支/commit/diff/恢复 | 阶段 A–D | 能保护已有更改并回滚单个错误 |
| Debugging (`pdb`/VS Code) | 阶段 A–C | 能设断点，检查 Tensor 中间值 |
| Shape / dtype / device 诊断 | 阶段 C | 能预测 reshape/broadcast 输出 |
| 单元测试与 gradcheck | 阶段 C–D | 能发现错误实现，不只检查能否运行 |
| 训练日志 / Checkpoint | 阶段 D | 能恢复训练并评估 |
| 配置与命令行 | 阶段 A–E | 参数、代码、实验相互分离 |
| AMP、显存与性能分析 | 阶段 D–E | 能用测量论证优化效果 |
| 论文公式→代码映射 | 阶段 E–J | 能说清函数对应的算法步骤 |
| 文档/实验复现 | 全程 | 别人能从 README 理解运行条件 |

**必要的数学能力随课补齐**：线性代数、概率/统计、微积分/链式法则、优化、信息论、常微分方程（在 Flow Matching 前）。不要求先完成整本数学书。

## 15. 每一类任务的通用模板

### 15.1 导师布置任务时应包含

- 任务名称、任务意义与本节目标；明确练习培养哪项能力、为什么现在练、对应什么实际用途。
- 已有前置知识（哪些已验收、哪些本节介绍）。
- 允许修改的文件（明确路径）；不允许触碰哪些已有文件。
- 输入、输出、Tensor Shape、dtype/device（适用时）。
- 函数接口或伪代码；**不提供完整函数体**。
- 少量核心验收条件，重点是运行成功且正确体现本章内容；边界/跨设备验证仅在与本章核心目标直接相关时要求，否则为可选改进。
- 有理解疑点时直接提出针对性问题；学生可在聊天或注释中回答，不强制预测注释或重复完整手算。
- 当学生卡住时先参考的官方文档/API 名称。

### 15.2 学生提交代码后，导师 Review 模板

1. **确认事实**：读到的文件、命令/输出、无法验证的事项。
2. **结果**：通过 / 基本通过需修改 / 暂不通过。
3. **做对的地方**：准确指出行为与理由（不要无原则表扬）。
4. **最重要的问题**：按正确性 → 逻辑 → Shape → API → 工程风格排序。
5. **提示**：解释原理、位置、检查方法及局部修改策略，不默认给可直接运行的局部代码。
6. **核心问题由学生修正并验证**；非核心问题作为可选建议。需要确认理解时直接提问，接受聊天或注释回答，不机械附加新关卡。
7. **更新 `PROGRESS.md`**，仅记录实际完成的项目/验收/问题。

### 15.3 防止“形式上手写、实际上不会”的设计

出现具体理解疑点时，按需选择有效方式；以下不是每项任务必做的验收清单，不强制注释预测、整张量手算或反复变体：

- 给一个数值与 Shape 完全不同的小变体。
- 让学生先预测输出再执行。
- 把一个操作替换为不等价 API，让学生解释错误。
- 让学生画调用链和输入/输出结构。
- 对比可运行但结果错误的实现与正确实现。
- 从空白文件复写“核心 10–30 行”的思路（不要求每次复写整项目）。
- 让学生解释一个 API 的替代方案与副作用。

## 16. 实验验证习惯与算法质量标准

任何算法实现都优先测试下列 8 点：

1. **Shape**：输入/输出满足约定；不正确时尽早报错。
2. **dtype/device**：CPU/GPU、float32/bfloat16、整数标签是否正确。
3. **数值**：无 NaN/Inf，loss 有意义；对概率分布做总和/范围检查。
4. **梯度**：需要更新的参数有梯度；不应更新的参数不参与计算图。
5. **过拟合**：若适用，能拟合单个 batch（发现训练管线问题）。
6. **种子与复现**：固定必要的随机种子，记录版本；知道并非任何 CUDA 算子都严格确定。
7. **边界条件**：例如 t=0、最后一步、空输入、不足一个 batch、mask 全零等。
8. **基准对照**：手写实现先与简单真值/官方算子对齐，再谈性能。

**禁止“loss 降了所以算法一定正确”的判断**：必须用独立性质或可计算真值验证。

## 17. 进度状态、优先级与跨会话交接

### 17.1 三种知识状态不能混淆

- `NOT_STARTED`：尚未系统讲授。
- `EXPLAINED`：讲授/演示过，但没有验收学生独立实现。
- `IN_PROGRESS`：学生已经开始亲手编写、未完成验收。
- `PASSED`：已完成代码、解释、验证。
- `NEEDS_REVIEW`：曾完成但核心理解存在缺口，需要复习。
- `OPTIONAL`：研究方向选修，暂不影响主线推进。

状态以 `PROGRESS.md` 为准，勿自行“乐观升级”。

### 17.2 每次课程结束必须交接的内容

- 日期、课程 ID、学习了什么。
- 学生**亲手编写或修改**的文件及简述，不要贴整段代码。
- 验收证据（测试命令/结果、Git commit 可选）。
- 现在**确实掌握**与**仍不清楚**的点。
- 下次进入课程前复习的 1–3 个问题。
- 下一个**具体最小任务**（可在 20–60 分钟内完成的粒度，不是整个里程碑）。
- 遗留错误/环境变更/需要注意的本地文件。

### 17.3 每隔一个阶段做一次 Review

- 回顾从头能否独立写最小模块。
- 整理易错知识清单。
- 根据真实掌握水平调整课次，不追求数量。
- 提取 reusable notes、测试和小项目，不复制 AI 成品。

### 17.4 自动记录与跨设备恢复（2026-10-12 学生确认）

- **验收后自动记录**：立即维护 `PROGRESS.md` 的当前状态、证据、待复习点和下一项最小任务，追加实际 Session Log，保留历史；此项已有持续授权，无需反复确认。仅当课程路线或长期规则改变时更新本计划及 `AGENTS.md`，实际进度统一存于 `PROGRESS.md`。
- **每次开课先读进度**：无论是否新会话、是否同一设备，都重新检查磁盘上的最新进度与相关代码；首先简短说明从哪里继续。指导文档中的建立时快照、旧聊天及 Session 模板不能替代最新实际记录。
- **跨设备操作顺序**：当前设备学习结束 → 将学生作业与更新的文档一起 commit、push → 另一设备开课前 pull → 导师重新读取记录并继续。自动维护文档不代表自动运行 Git 同步命令；学生提出执行同步的请求时再处理。
- **检查范围**：本地 Git 状态只能说明当前工作区情况，不能证明远端没有新提交。发现同步冲突时保留双方实际记录，依据代码与验收证据核对；不能仅按更晚日期或更高课次覆盖进度。
- **设备环境**：作业路径使用仓库相对路径；Windows/Linux、CPU/CUDA 等环境分别记录。切换设备后重新核实当前解释器和所需设备，不能把另一设备的成功运行当成本机验证。

## 18. 本课程的当前学习状态与下一步（建立计划当天）

### 已经讲解，但未验收

- 项目与模块的初步区别；一个学习项目的基础目录。
- Conda 环境的创建、激活命令含义；Conda / pip 的基本定位；使用 PyTorch CUDA wheel 的概念。
- PyTorch `import torch`、Tensor 创建、`shape`、`dtype`、`device`、均值、二维索引、切片、`.to("cuda")`。
- GPT 学习的最初目标与六阶段简版路线（现在扩为本计划）。

### 尚未获得证据

- `lessons/01_tensor.py` 是否存在/已运行。
- `dllab` 的 `torch.cuda.is_available()` 是否返回 True。
- 学生是否真正理解 `mean(dim=0)` vs `mean(dim=1)`、`x.to('cuda')` 是否原地、二维索引。
- Python 高级语法、Autograd、训练循环是否可独立编写。

### **第一项待办：C01 / Lesson 01（不要越级）**

学生在 `~/projects/DeepLearningLab/lessons/01_tensor.py` **自己手写**：

1. 创建值为 `[[1.,2.,3.],[4.,5.,6.]]` 的 2×3 浮点 Tensor。
2. 打印张量及 `.shape`、`.dtype`、`.device`。
3. 索引第一行与第二列并解释结果 Shape。
4. 计算按列的平均值，并解释 `mean(dim=0)` 与 `mean(dim=1)` 的区别。
5. 把 Tensor 移动到 CUDA；清楚区分 `x.to('cuda')` 返回新 Tensor 与是否修改 `x` 变量。
6. 自己给一个不同 Shape 的测试例子，并预测结果。

导师的职责是：先检查是否已有文件，**不要直接替学生创建作业答案**；讲必要语法、给小提示、按验收规则 Review。成功后再考虑按进度进入 C02 或穿插 A 系列的 Python 基础补课。

### 初期教学策略（重要）

“第一个作业 C01”只是自然入口；若学生在写代码时卡在 Python 的 `import`、函数、列表、变量绑定或模块组织，应回到 A01–A07 插入微课，**不是重复整套课程或跳过障碍**。此后按数学需求穿插 B 系列。

2026-10-12 学生已明确确认第 3 节的实践主线安排：继续以 C 及后续内容组织教学，A/B 按前置需求穿插、联系并验收；不将本节建立计划当天的历史快照作为当前待办。

## 19. 课程毕业标准（可检验，不是口号）

通过课程后，学生应该能在有限查阅官方文档的前提下：

- 90 分钟内把小神经网络的数学公式翻译为结构清晰的 PyTorch 模型类和训练循环（不要求固定速度）。
- 不看现成源码，独立重建 Transformer Block 的重要组件，写 Shape / causal mask / gradient 测试。
- 维护自己的 mini-GPT：配置、训练、评价、生成、ckpt、复现均可自行调试。
- 独立实现 toy DDPM 与 toy Flow Matching，解释 loss 与采样器差别，正确比较公式与 Shape。
- 遇到新模型能够找到核心计算图、阅读训练入口并给出最小可验证改动。
- 能清楚说出“我目前不理解什么、我怎样验证自己的假设”，而不是把模型输出当作事实。

毕业不是课程终点，之后可以以“读论文 → 推导 → 实现最小模块 → 写测试 → 对照开源实现”不断循环。

## 20. 官方资料与参考项目（学习时按需使用，非必须立刻阅读）

- Python Tutorial：https://docs.python.org/3/tutorial/
- Python 标准库：https://docs.python.org/3/library/
- Packaging User Guide：https://packaging.python.org/
- NumPy：https://numpy.org/doc/stable/
- PyTorch Tutorials：https://docs.pytorch.org/tutorials/
- PyTorch API：https://docs.pytorch.org/docs/stable/index.html
- micrograd：https://github.com/karpathy/micrograd
- nanoGPT：https://github.com/karpathy/nanoGPT
- Attention Is All You Need：https://arxiv.org/abs/1706.03762
- DDPM：https://arxiv.org/abs/2006.11239
- DDIM：https://arxiv.org/abs/2010.02502
- Flow Matching for Generative Modeling：https://arxiv.org/abs/2210.02747
- Rectified Flow：https://arxiv.org/abs/2209.03003
- Codex AGENTS.md 参考：https://developers.openai.com/cookbook/examples/gpt-5/codex_prompting_guide
- Agent Skills 参考：https://developers.openai.com/api/docs/guides/tools-skills

原则：优先官方文档、原始论文、官方或作者开源仓库；博客/视频用于补充直觉，遇到版本变化时核实当前 API，不把旧文章当作唯一事实。

## 21. 使用方式：如何让新 Agent 接手

将本包文件放到 `~/projects/DeepLearningLab/` 的相应相对路径后，从此目录启动 Codex。你可以对新 Agent 说：

> 请先阅读项目根目录的 `AGENTS.md`、`LEARNING_PLAN.md`、`PROGRESS.md`，并按需要使用 `deep-learning-mentor` skill。你是我的编程导师，不是代写者。请只读检查 `lessons/` 与环境；我希望继续做第一节 Tensor 作业，先询问我已经写了哪些内容，不要自动生成完整解答。之后每次学习完成再根据证据更新 `PROGRESS.md`。

如果切换回网页 ChatGPT，只需上传 `LEARNING_PLAN.md`、`PROGRESS.md` 和必要代码，就可以较可靠地恢复到当前课程状态。**任何 Agent 都可能丢失会话上下文，磁盘上的进度与代码才是持续交接的依据。**

---

*维护原则：课程大纲低频更新，实际进度高频更新。若新增前沿算法，在阶段 I 扩展即可；不为追逐新论文而破坏 Python/PyTorch/Transformer/Diffusion/Flow Matching 核心链路。*
