# 作者资源与课程映射

核对日期：2026-10-28。此表是教材入口，不是待看完的必修列表，也不是学生已掌握的证明。课程组合是本项目的教学判断，不声称与作者课次一一对应。

## 1. nanoGPT目标和代码入口

- [nanoGPT README](https://github.com/karpathy/nanoGPT)：已读项目定位、字符级入门、训练/推理及其教育资源链接。目标取小模型闭环，不要求复现GPT-2训练规模、硬件或README中的耗时。上游目前标注其为旧项目并指向nanochat；学生指定的nanoGPT学习目标保留，不因此自动改学完整聊天系统。
- [model.py](https://github.com/karpathy/nanoGPT/blob/master/model.py)：已核对CausalSelfAttention、MLP、Block、GPT输入/损失路径。E阶段逐步对应[B,T]→[B,T,C]→logits[B,T,V]；先解释公式、Shape和因果性，再对照源码。
- [train.py](https://github.com/karpathy/nanoGPT/blob/master/train.py)：已核对get_batch、评估、检查点、累积、缩放和裁剪的位置。D08/D09对应训练状态和有效更新；注意参考源码的检查点字段不等于本项目“完整RNG精确接续”练习的全部要求。
- sample.py与data目录在README中作为采样/数据入口出现，本轮未逐行审阅其实现；到E阶段对应课再读取。

## 2. 沿作者教育链接继续学习

| 一手资料 | 本轮实际读取范围 | 对应课程/使用方式 |
|---|---|---|
| [Zero to Hero课程页](https://karpathy.ai/zero-to-hero.html) | 已读课程简介、先修与各讲主题 | 用于从反向传播、字符语言模型到GPT的主题组织；不要求重新学完已通过的MLP/BN |
| [课程仓库](https://github.com/karpathy/nn-zero-to-hero) | 已读README、Notebook/练习链接与GPT/Tokenizer说明 | 按主题选材料，不将课程播放时长当成本项目课时 |
| [makemore](https://github.com/karpathy/makemore) | 已读项目介绍；未审完所有Notebook | E01–E03的bigram、上下文预测与学习到的表示 |
| [minGPT](https://github.com/karpathy/minGPT) | 已读README的教育定位与model/trainer/tokenizer边界 | 用作较易解释的对照；同样是旧项目，API按当前环境校对 |
| [Backprop练习Notebook](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part4_backprop.ipynb) | 已读取原始Notebook中的输入构造与梯度比较工具，并结合课程说明核对练习目的；未逐单元审阅或执行 | 数学缺口出现时选一个局部导数/张量梯度验证，不整份补做为进入GPT门槛 |
| [minbpe练习](https://github.com/karpathy/minbpe/blob/master/exercise.md) | 已读分阶段练习，包含BasicTokenizer到更完整编码器的递进 | 字符级mini-GPT完成后按需扩展BPE；不先要求复刻GPT-4 tokenizer |
| [minbpe文字讲义](https://github.com/karpathy/minbpe/blob/master/lecture.md) | 已查看讲义入口与tokenization说明，未执行参考代码 | BPE专题备课入口，替代一次性堆砌编码细节 |

课程README说明其他练习可见视频描述。本轮只读取了视频页面/课程说明，没有完整观看所有视频，也未遍历所有视频描述中的资源；不能写成“全部视频/作业已审阅”。讲义、代码、练习含参考答案时，默认只提供题意、框架和提示。

## 3. Slides与访问边界

作者[主页](https://karpathy.ai/)链接了[State of GPT slides](https://karpathy.ai/stateofgpt.pdf)。本轮打开后未获得可核实的PDF正文，另一次源文件访问也未成功，因此**未将它计为已读slides**，不据其未知内容新增必修要求。已读的nanoGPT/Zero to Hero入口并未提供一套统一对应每课的slides目录；本项目以可核实的课程说明、Notebook、源码和练习为主，不能把自行制作的讲义称作作者slides。

若未来需要某份slides的具体图表，先重新核实可访问版本，再阅读对应页。理论原始依据可按主题查[Attention Is All You Need](https://arxiv.org/abs/1706.03762)；此链接是论文，不标为slides，本轮也不冒称重新全文审阅。

## 4. 选材与使用规则

每次只选一段与当前主题直接相关的一手资源，记录“资源→本课问题→学生代码产物→验证方法”。API查当前官方文档，资源里的旧安装命令、性能数字和可选优化不能直接搬到本机。

首个mini-GPT的必做闭环：文本/ID与错位目标→上下文表示→因果注意力→多头Block→训练/验证→自回归生成→保存恢复→源码映射。BPE、高级位置编码、KV Cache、FlashAttention、DDP及大规模数据是后续扩展；它们的存在不使一个正确的小型字符GPT“不合格”。
