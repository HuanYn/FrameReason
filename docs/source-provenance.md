# 数据、结果与代码出处

## 外部来源

- 数据集：[CLEVRER 官方项目](https://clevrer.csail.mit.edu/)，视频问答标注与视频来自 CLEVRER。数据许可见官方 [`README.txt`](https://data.csail.mit.edu/clevrer/README.txt)，为 CC0。本仓库只附带少量展示视频及参考题，完整数据应从官方来源取得。
- 基座：[Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)，历史 revision 为 `ebb281ec70b05090aa6165b016eac8ec08e71b17`。权重未包含在仓库中，使用基座需同时遵守其模型许可。
- 训练框架：[modelscope/ms-swift](https://github.com/modelscope/ms-swift)。本项目贡献是 CLEVRER 输入/目标适配、规则奖励、固定评测与结果诊断；不将 ms-swift、Qwen、LoRA 或 GRPO 的原有方法记作自行发明。

## 本仓库公开的结果证据

| 文件 | 来源与用途 |
| --- | --- |
| [`results/predictions.jsonl`](../results/predictions.jsonl) | 从三份各 800 条历史预测中逐项导出，共 800 行、每行三模型。`response` 保留原始模型输出；`answer` 是原始标签内的文本；`answer_correct` 来自当时的逐题评分账本。 |
| [`results/subset800.jsonl`](../results/subset800.jsonl) | 从历史固定测试集 identity records 与原始预测交叉核对，包含视频 ID、问题 ID、题型、record ID 和 ordinal。 |
| [`results/demo-references.jsonl`](../results/demo-references.jsonl) | 10 道已保存的正式测试记录中的官方监督目标和公开输入文本。没有把模型预测反推成官方真值。 |
| [`results/training-groups.jsonl`](../results/training-groups.jsonl) | 从原始 completion/scalar 日志对齐的 3000 个训练组，每组保留 4 个奖励、4 个优势与该步梯度范数。不是 3000 个独立视频。 |
| [`results/metrics.json`](../results/metrics.json) | 答案计数、题型计数、配对统计与奖励组统计由以上记录重计；格式、完整程序和联合成功等附加指标按历史 metrics 摘录。 |
| [`results/provenance.json`](../results/provenance.json) | 原始存档与公开派生文件 SHA-256、模型 revision、历史版本和 adapter manifest 摘要。只发布摘要，不包含权重。 |
| [`results/historical-config.json`](../results/historical-config.json) | 已完成主线的模型、LoRA、SFT、GRPO、解码、硬件和奖励配置；敏感部署字段已删除。 |

导出采用字段白名单，而不是将整份机器日志替换几个字符串后上传。不公开密码、API key、服务器地址、GPU UUID、个人目录、完整模型/优化器状态或完整输入帧。`results/export_local_archives.py` 是维护者使用的离线导出工具，需要自行提供历史文件；普通使用者无需运行它，也不会由它联网获取缺失文件。

### 正确性标签如何取得

Base、SFT 的完整 failure ledger 与 metrics 留存在本地归档；GRPO 的完整 failure ledger 精简导出保留了逐题答案是否正确。失败账本是多标准的：答案正确但程序错误也可能出现在其中。导出时使用明确的 `answer_correct` 字段；没有出现在完整失败账本的题目，按历史评分协议视为各项成功。**不能把“有 failure 行”直接当成“答案错误”。**

导出核对三模型 800 条 identity/ordinal 完全一致，并核对原始预测 SHA 与历史 metrics 中声明的摘要一致。GRPO 共有 195 条多标准失败记录，其中只有 194 道答案错误，这也解释了为什么 800 − 195 不等于最终的 606。

默认 CPU 验证是历史标签重计数，加上 10 道有官方参考的答案重评分，不假称重新独立重评分完整 800 道官方标签。完整参考标签准备方式与可选重评分入口见[复现范围](reproduction-scope.md)。

### 同分组统计如何取得

原始 `completions.jsonl` 有 3001 行，最后一行与倒数第二行完全相同。只在派生视图中排除该末尾重复，原日志不改动。逐组奖励四舍五入至小数点后 6 位后比较，得到 2659 个同分组；同分奖励为 1.0、0.1、0.5、0.6 的组数分别为 2110、457、87、5。对应组的记录优势全部为零；scalar 日志中另核对出 2659 个零梯度范数步。

这些计数可用公开文件复核。没有同时发布 12000 条训练生成内容，因此公开验证不声称再次核验全部训练答案/程序的官方 GT。历史完整逐回答奖励重算与本次公开的数值日志重计是两种不同的证据。

## Demo 的真实身份

Demo 回答是 **保存的历史模型输出**，不是实时在线推理，不是为展示编写的回答。视频取自与题目 `video_id` 对应的 CLEVRER 官方 validation 视频；兼容视频/动图是由这些真实视频转码得到的展示资源，不是模型生成视频。

10 道题为解释差异而选，不用于报告整体准确率：先取每类最早的两道 Base/SFT 答案不同的题，共 8 道；再补最早的一道 GRPO 纠正和一道 GRPO 丢失。后两道按结果挑选是刻意且公开的，不能把精选展示当作随机样本。全量结论只能来自固定 800 题。

历史模型实际收到的是八张抽帧图。Demo 里能播放完整视频，并不代表历史推理读取了所有帧；本次公开没有重验八张输入 JPEG 的全部字节。中文题意用于人类阅读，原英文题干与模型原始回答分别保留。

## 版本与不可复现的部分

五个保留算法实现的 Python 模块在公开时仅将 CRLF 换行规范化为 LF。原存档字节哈希与公开文件哈希分别保存在 [`configs/provenance.json`](../configs/provenance.json)，避免把换行变化误认为来源不明，也避免 Windows 本地检查通过而新克隆检查失败。训练和推理的便携启动器是本次新增封装，不能冒充原机器调度代码。

历史环境为 Python 3.11.11、torch 2.6.0+cu124、transformers 4.57.3、ms-swift 4.5.2、peft 0.19.1、trl 0.29.1。版本是历史运行记录，不代表任意当前依赖组合都能无修改运行。

`provenance.json` 中训练、评测协议、恢复执行的 commit 是历史工作树身份标识；它们不一定是上游公共仓库可以直接检出的 commit。本仓库发布的是经过整理的最小实现，不要求使用者访问私有实验服务器才能启动。

历史测试 JSONL 带有机器上的绝对媒体路径，所以重新准备数据后文件 SHA 不会自然相同。准确复用科学比较所需的是相同题目键、标签、视频与采帧规则、模型/adapter、解码和评分协议；仅有一个旧 SHA 并不能补足缺失的权重或输入帧。
