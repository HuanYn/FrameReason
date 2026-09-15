# 来源与许可

## 代码

FrameReason 的任务适配代码从本项目使用的 ms-swift 开发工作树中提取。上游框架是 [modelscope/ms-swift](https://github.com/modelscope/ms-swift)，采用 Apache License 2.0；本仓库保留该许可证。`framereason/` 中的数据转换、均匀抽帧、输出解析和规则奖励代码保留原算法语义，便携化改动与来源哈希见 [代码来源](docs/source-provenance.md)。训练器本身通过依赖安装，没有复制整个 ms-swift 框架。

## CLEVRER 数据与演示素材

来源：[CLEVRER 官方项目页](https://clevrer.csail.mit.edu/) 和 [官方数据 README](https://data.csail.mit.edu/clevrer/README.txt)。官方 README 的 License 部分明确将数据集发布为 CC0。数据作者为 Kexin Yi、Chuang Gan、Yunzhu Li、Pushmeet Kohli、Jiajun Wu、Antonio Torralba 和 Joshua B. Tenenbaum；论文发表于 ICLR 2020。

`demo/media/` 仅包括十段官方 validation 视频的 VP8/WebM 播放副本与首帧预览，以及两个案例的 GIF 副本。转码保留视频顺序、画幅和时长；GIF 使用有限调色板。它们不是 FrameReason 生成的视频。素材与原文件的哈希保存在演示素材清单中。CLEVRER 的许可不由本仓库的代码许可证替代。

问题及标准答案来自 CLEVRER 官方标注。`results/` 中的模型回答来自本项目历史生成记录；中文演示译文只用于阅读，未作为原实验推理输入。

## 模型

基础模型来自 [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)。官方模型卡标注 Apache 2.0。本仓库不重新分发基座权重、SFT/GRPO 适配器或 tokenizer；使用者应从官方来源准备模型并遵守相应条款。

## 外部软件

运行时依赖 PyTorch、Transformers、PEFT、TRL、ms-swift 和 FFmpeg 等项目，各自遵循其原始许可证。版本记录不构成对依赖软件的重新授权。本仓库没有教师 API 密钥、云平台登录信息或外部服务凭据。
