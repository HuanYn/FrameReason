# 从历史回放到新训练

以下命令以 Linux shell 为例。仓库根目录是命令的工作目录；`FR_ROOT`、`FR_MODEL` 指向使用者自己的数据盘和本地模型快照。这里的路径是示例，没有绑定原项目的服务器。

## 1. 先验证不需要 GPU 的部分

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python -m unittest discover -s results -p "test_*.py" -v
python results/verify_results.py
```

前两条用于安装本地任务包和测试；结果复核脚本重新计算 800 题历史正确性标记的总数、配对变化和 3000 组奖励的同分比例，并独立评分随仓库提供的十题官方参考。它不把模型自己的答案当成缺失的官方答案。

静态 Demo：

```bash
python -m http.server 8766 --bind 127.0.0.1 --directory demo
```

打开 `http://127.0.0.1:8766/`。如果该端口已被其他本地服务占用，换成空闲端口。可选的无依赖素材检查为 `node demo/check.cjs`；浏览器检查方式见该脚本的说明。

## 2. 准备官方数据

在 [CLEVRER 官方页面](https://clevrer.csail.mit.edu/)下载训练和验证视频，以及 Questions and Answers。下面使用的是含 `questions` 的问题标注 JSON，不是只含物体轨迹的场景标注 JSON。

```bash
export FR_ROOT=/data/framereason
export FR_MODEL=/data/models/Qwen3-VL-4B-Instruct
```

将视频放到如下目录。官方压缩包按多个子文件夹存储视频；需要将对应文件整理或链接到这里。保留原始文件，不覆盖同名但内容不同的文件。

```text
/data/framereason/
  raw/questions/train.json
  raw/questions/validation.json
  raw/videos/train/video_00000.mp4
  raw/videos/train/...
  raw/videos/validation/video_10000.mp4
  raw/videos/validation/...
```

如果采用其他目录布局，可使用转换器的 `--train-video-template` 和 `--validation-video-template` 指定相对模板。`--train-annotations` 和 `--validation-annotations` 均支持多个输入文件。帧根目录必须预先存在且不能是符号链接；若使用视频链接，解析后的目标仍须位于声明的视频根目录内。

```bash
mkdir -p "$FR_ROOT/frames"

python -m framereason.convert_clevrer \
  --train-annotations "$FR_ROOT/raw/questions/train.json" \
  --validation-annotations "$FR_ROOT/raw/questions/validation.json" \
  --video-root "$FR_ROOT/raw/videos" \
  --frames-root "$FR_ROOT/frames" \
  --output-dir "$FR_ROOT/prepared" \
  --media-mode frames8 --seed 42 --dev-ratio 0.1 \
  --train-per-type 3000 --dev-per-type 500 --test-per-type 1000 \
  --require-full-caps --allow-missing-media

python -m framereason.extract_frames \
  --manifest "$FR_ROOT/prepared/manifest.jsonl" \
  --frames-root "$FR_ROOT/frames" \
  --receipt "$FR_ROOT/prepared/frame_receipt.json" \
  --frame-count 8

python results/select_subset.py \
  --input "$FR_ROOT/prepared/grpo_test.jsonl" \
  --output "$FR_ROOT/prepared/compact800.jsonl"
```

`--allow-missing-media` 用于先生成帧路径和抽帧清单，不是允许训练时缺少图片。抽帧阶段需要原视频及 FFmpeg，真正训练前仍会检查八张图片。`requirements-media.txt` 提供可选的 FFmpeg Python 包依赖。

最后一条按发布的精确题目键恢复 800 题成员和顺序。`python -m framereason.subset` 另保留原始哈希抽样算法，用于研究抽样过程。新生成 JSONL 的绝对图片路径不同，文件字节哈希自然可能不同，不应通过改写原历史哈希掩盖差异。

如需对已发布的所有历史回答独立重评分：

```bash
python results/verify_results.py \
  --reference-jsonl "$FR_ROOT/prepared/compact800.jsonl"
```

## 3. 配置训练环境

历史环境为 Linux、Python 3.11.11 和单张 RTX 3090 24GB。使用独立虚拟环境，在数据盘保存依赖缓存、模型和训练输出。

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-train.txt
```

这只是主要依赖的版本配方。历史训练使用经过项目开发的 ms-swift 工作树，公开最小包没有复制整个工作树，因此尚未证明上述干净安装与原运行逐步等价。运行环境、数据、程序和随机性都可能影响新结果。不要在未做环境验证时承诺复现到同一分数。

从 [官方模型页](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)准备 revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`，将 `FR_MODEL` 指向已有的本地快照目录。执行器只使用本地模型，默认关闭在线模型访问和 W&B。

## 4. SFT 与开发集选模

下面先打印训练计划，不会启动 GPU：

```bash
python -m framereason.launch sft \
  --model "$FR_MODEL" --data "$FR_ROOT/prepared" \
  --output "$FR_ROOT/runs/sft750"
```

参数来自 `configs/sft.json`。真正执行时，在同一命令后追加 `--execute --gpu <编号> --expected-gpu-uuid <实际UUID> --cache-root "$FR_ROOT/cache" --max-hours <本次时限>`，尖括号部分必须替换为实际值。执行前检查 GPU UUID、显存占用小于 500 MiB 且无计算进程。文件锁只协调使用同一缓存目录的本项目启动器，不能替代共享服务器的资源调度。

完成训练后，对候选 checkpoint 700 和 750 分别推理和评分。下例展示 700；750 使用不同的适配器、预测文件与评分文件名。

```bash
python -m framereason.infer \
  --model "$FR_MODEL" --adapter "$FR_ROOT/runs/sft750/checkpoint-700" \
  --dataset "$FR_ROOT/prepared/grpo_dev.jsonl" \
  --output "$FR_ROOT/evaluation/sft700-dev.jsonl"
```

推理同样默认只展示计划。确认后追加 `--execute --gpu <编号> --expected-gpu-uuid <实际UUID> --cache-root "$FR_ROOT/cache"` 执行。必须等全部预测完成后才能评分，缺失或重复题目会被拒绝。

```bash
python -m framereason.evaluate \
  --dataset "$FR_ROOT/prepared/grpo_dev.jsonl" \
  --predictions "$FR_ROOT/evaluation/sft700-dev.jsonl" \
  --output "$FR_ROOT/evaluation/sft700-dev-metrics.json"

python -m framereason.select_checkpoint \
  --dev-dataset "$FR_ROOT/prepared/grpo_dev.jsonl" \
  --candidate 700 "$FR_ROOT/runs/sft750/checkpoint-700" "$FR_ROOT/evaluation/sft700-dev-metrics.json" \
  --candidate 750 "$FR_ROOT/runs/sft750/checkpoint-750" "$FR_ROOT/evaluation/sft750-dev-metrics.json" \
  --output "$FR_ROOT/evaluation/selected-sft.json"
```

先完成两组候选推理与评分，再运行选模命令。历史 winner 是 700，新训练可能选择不同版本；必须记录新选择，不能将其直接标记为历史 SFT700。

## 5. GRPO 与固定测试

下例按历史 SFT700 初始化。新运行使用自己在 dev 选出的适配器时，应同步修改参数和实验名称。

```bash
python -m framereason.launch grpo \
  --model "$FR_MODEL" --data "$FR_ROOT/prepared" \
  --adapter "$FR_ROOT/runs/sft750/checkpoint-700" \
  --output "$FR_ROOT/runs/grpo3000"
```

它默认打印计划，显式执行方式与 SFT 相同。`configs/grpo.json` 固定四候选、temperature 0.9、beta 0、320 token 和 3000 步。不要根据最终测试结果挑选 checkpoint。

完成后，分别对同一 `compact800.jsonl` 使用 Base、选定 SFT 和 GRPO3000 生成预测。Base 不传 `--adapter`，SFT 只传 SFT 适配器，GRPO 只传其最终 actor 适配器。三者各生成独立文件，再用 `framereason.evaluate` 对同一官方参考评分。

```bash
python -m framereason.infer \
  --model "$FR_MODEL" --adapter "$FR_ROOT/runs/grpo3000/checkpoint-3000" \
  --dataset "$FR_ROOT/prepared/compact800.jsonl" \
  --output "$FR_ROOT/evaluation/grpo3000-test.jsonl"

# 在显式执行并完整生成后，进行 CPU 评分：
python -m framereason.evaluate \
  --dataset "$FR_ROOT/prepared/compact800.jsonl" \
  --predictions "$FR_ROOT/evaluation/grpo3000-test.jsonl" \
  --output "$FR_ROOT/evaluation/grpo3000-test-metrics.json"
```

训练、推理和评分输出使用新目录或新文件，拒绝直接覆盖已有结果。原项目的权重与优化器状态没有随仓库发布，因此本流程从用户自行准备的模型开始，而不是恢复原服务器上的训练进程。
