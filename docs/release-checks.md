# 最小发布验证

本次整理没有启动 GPU、教师 API 或新的模型评测。检查的是公开代码、已保存的结果和 Demo，不是重新训练后的性能。

| 检查 | 结果 | 范围 |
| --- | --- | --- |
| 核心 CPU 回归 | 23 项通过 | 转换、抽帧索引、公开输入隔离、解析、奖励、评分、参数、显式 GPU 检查的模拟测试 |
| 结果 CPU 回归 | 6 项通过 | 答案规范化、历史计数、十题参考重评分、篡改与重复记录拒绝、精确子集提取 |
| 历史记录重计 | PASS_WITH_DECLARED_SCOPE | 800 题、754 视频；三模型 343/609/606；GRPO 59 纠正/62 丢失；2659/3000 同分 |
| Demo 静态检查 | 通过 | 十视频、三十回答、二十二个媒体哈希，共 6,012,650 字节 |
| Demo 浏览器检查 | 安装版 Microsoft Edge 通过 | 十段 WebM 播放、两个 GIF、题型筛选、标准答案、320 至 1360 像素宽度；无外部 HTTP 请求 |
| Git 归档检查 | 通过 | 从暂存树导出的全新目录重复运行二十九项 CPU 测试、结果重计与 Demo 静态检查，验证 LF 换行下的来源哈希 |
| Python wheel | 构建与隔离导入通过 | 无训练依赖安装，验证包内 SFT/GRPO 配置可读取；不代表 GPU 环境已验证 |
| 发布范围检查 | 通过 | 未发现凭据、私有服务器路径、模型权重、完整数据或机器缓存；相对文档链接有效 |

复核命令：

```bash
python -m unittest discover -s tests -v
python -m unittest discover -s results -p "test_*.py" -v
python results/verify_results.py
node demo/check.cjs
```

可选浏览器检查使用 `node demo/check.cjs --browser`，需预先安装 Playwright 和可用浏览器，必要时指定 `BROWSER_EXECUTABLE`。脚本不会自动安装浏览器。

完整 800 题的官方 ground truth 没有随结果包提供，默认重计不等于独立官方标签重评分。使用者提供相同题目键的完整官方参考后，可额外运行 `python results/verify_results.py --reference-jsonl <path>`。历史训练使用开发版框架，干净依赖安装及完整 GPU 训练尚未在这个最小发布中重跑。
