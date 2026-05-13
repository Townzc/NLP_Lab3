# NLP Lab3: Seq2Seq 德英翻译

本仓库完成实验三“Seq2Seq模型实现文本翻译（德英翻译）”。代码基于 MindSpore 2.2，实现 Multi30K 数据下载、分词、词典构建、Seq2Seq + Bahdanau Attention 模型训练、推理、BLEU 评估，以及训练记录和注意力热力图保存。

## 华为云 ModelArts 运行

在 Notebook 中进入仓库目录后执行：

```bash
pip install -r requirements.txt
python src/seq2seq_translate.py --device_target CPU --epochs 10 --batch_size 128 --max_len 32
```

如果希望使用 Ascend，可以改为：

```bash
python src/seq2seq_translate.py --device_target Ascend --epochs 10 --batch_size 128 --max_len 32
```

脚本会自动下载并解压 Multi30K 数据集，默认输出到 `outputs/`：

- `outputs/training_history.csv`: 每轮训练损失、验证损失、困惑度
- `outputs/metrics.json`: BLEU、样例翻译、超参数等记录
- `outputs/predictions.txt`: 测试集样例翻译
- `outputs/training_curve.png`: 训练/验证损失曲线
- `outputs/attention_heatmap.png`: 注意力权重热力图
- `outputs/seq2seq.ckpt`: 最优验证损失模型权重

## 快速检查

可以先用较少 epoch 确认环境和数据流程：

```bash
python src/seq2seq_translate.py --device_target CPU --epochs 1 --batch_size 128 --bleu_limit 50
```

完整实验建议按指导书训练 10 轮，并在测试集上计算完整 BLEU。

## 报告

报告正文在 `report/实验报告三.md`，Word 完成版为 `实验报告三_完成版.docx`。实际在 ModelArts 运行后，可把 `outputs/` 中的曲线图、注意力图和 `metrics.json` 数值补充到最终提交版本中。
