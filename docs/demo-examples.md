# 视频与三模型回答 Demo

本演示保存了 **10 段 CLEVRER 视频、10 道题与 30 条历史模型回答**。它不运行在线推理，不请求教师 API，也不需要 GPU。英文问题、选项、原始输出、记录 ID 与答案评分保留不变；页面内的中文仅为展示译文。

在仓库根目录执行 `python -m http.server 8766 --bind 127.0.0.1`，打开 `http://127.0.0.1:8766/demo/`。也可直接打开 `demo/index.html`。GitHub 仓库文件页不会执行交互 HTML；下方 GIF 可直接预览。

## 案例一：GRPO 纠正了一个错误

![video_11060 的完整动图转码](../demo/media/video_11060.gif)

视频：`video_11060`，问题：`Q9`，原评测序号：`8`。

**How many collisions happen?** / 一共发生了多少次碰撞？

| 标准答案 | 原始 Qwen3-VL-4B | SFT700 | GRPO3000 |
|---|---|---|---|
| 1 | 2（错误） | 2（错误） | 1（正确） |

[下载兼容 WebM 视频](../demo/media/video_11060.webm)。这里只说明该题得到纠正，不说明 GRPO 整体优于 SFT。

## 案例二：GRPO 丢失了一个原本答对的结果

![video_10317 的完整动图转码](../demo/media/video_10317.gif)

视频：`video_10317`，问题：`Q1`，原评测序号：`0`。

**How many cylinders are moving?** / 有多少个圆柱体在运动？

| 标准答案 | 原始 Qwen3-VL-4B | SFT700 | GRPO3000 |
|---|---|---|---|
| 0 | 0（正确） | 0（正确） | 1（错误） |

[下载兼容 WebM 视频](../demo/media/video_10317.webm)。完整固定 800 题中，GRPO 相对 SFT 纠正 59 题、丢失 62 题，净少 3 题。

## 所有展示样例

选项题答案中的数字是**从 0 开始的选项 ID**，不是自然语言答案；`[]` 表示不选择任何选项。完整选项和原始输出见 [samples.json](../demo/samples.json) 与交互页面。

| 视频 / 问题 | 题型 | Base | SFT700 | GRPO3000 | 标准答案 |
|---|---|---|---|---|---|
| video_13559 / Q3 | 描述 | purple cylinder | cylinder | cylinder | cylinder |
| video_13945 / Q2 | 描述 | 4 | 3 | 3 | 3 |
| video_13513 / Q11 | 解释 | [0] | [] | [] | [] |
| video_10614 / Q11 | 解释 | [0,2] | [2,3] | [2,3] | [2,3] |
| video_11682 / Q13 | 预测 | [0] | [1] | [1] | [1] |
| video_14121 / Q11 | 预测 | [0,1] | [1] | [1] | [1] |
| video_12373 / Q14 | 反事实 | [2] | [0,2] | [0,2] | [0] |
| video_11961 / Q14 | 反事实 | [0,1,2,3] | [0,3] | [0,3] | [0,2,3] |
| video_11060 / Q9 | 描述 | 2 | 2 | 1 | 1 |
| video_10317 / Q1 | 描述 | 0 | 0 | 1 | 0 |

`video_13559 / Q3` 的 Base 回答包含正确形状，但多了颜色，不符合严格答案格式。这不是“模型一定看错了物体”的证据。

## 选样与结论边界

前 8 题是在完整 800 题的原始顺序中，每种题型选择 Base / SFT 答案不同的前 2 题；后 2 题分别是最早的 GRPO 纠正与退步。它们是有意挑选的解释案例，**不得用这 10 题的正确率代表项目性能**。

本演示对应原始 4B、SFT700、GRPO3000，不含 8B 教师或后续补强分支。完整固定 800 题结果为 343 / 800（42.875%）、609 / 800（76.125%）、606 / 800（75.750%），来自单 seed、自定义 validation 子集，不是官方隐藏测试榜单。程序字段仅按标注匹配，并未通过物理引擎执行验证。

## 媒体来源、编码与许可

视频与问题来自 [CLEVRER](https://clevrer.csail.mit.edu/)，作者为 Kexin Yi、Chuang Gan、Yunzhu Li、Pushmeet Kohli、Jiajun Wu、Antonio Torralba、Joshua B. Tenenbaum。官方[数据集 README](https://data.csail.mit.edu/clevrer/README.txt) 明确将数据集以 **CC0** 提供。展示媒体沿用该数据集许可，不适用仓库代码许可来重新授权。

每段视频均为 480 × 320、25 fps、128 帧、5.12 秒。WebM 为同一原始 MP4 的完整 VP8 转码，没有裁剪或降帧；JPEG 是第 0 帧。为控制仓库体积，只给上述两个案例附加 GIF 备用播放，GIF 有 256 色量化。它们不是模型生成视频。原片 SHA256、转码 SHA256 和回答记录 ID 均保留在 `samples.json`；未附原始 MP4、完整数据集、模型权重或当时输入模型的 8 张 JPEG。

## 本地检查

运行 `node demo/check.cjs` 校验 10 个样例、30 条回答、媒体哈希、预期评分与公开数据边界。运行 `node demo/check.cjs --browser` 加上浏览器检查；该可选项需要本机已安装 `playwright`，可通过 `BROWSER_EXECUTABLE` 指定已有 Chromium / Chrome / Edge。不会自动安装浏览器、下载媒体或启动模型。

`export-archive.cjs` 是维护者用的白名单导出工具，输入一个已有本地归档目录，不是普通用户运行 Demo 的必需步骤。仓库已经附有导出结果，无需访问原实验服务器。
