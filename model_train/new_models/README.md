# new_models

本目录包含用于“人员识别（闭集）+ 跨域评估”的 1D 模型与（可选）WGAN 数据增强代码。所有新增代码均只放在该目录下。

## 数据输入

这些模型使用 MATLAB 事件检测输出的 `footstep_feat`（每条样本 1500 点）作为 1D 输入。

目录结构要求：

```
<interim_root>/<dataset>/<subset>/Pxx.mat
```

其中 `Pxx.mat` 内包含变量 `footstep_feat`，shape 为 `(N_events, 1500)`。

示例（服务器）：

```
/root/毕设/Terra-main/people_database/interim_matlab/A3/A3_1/P1.mat
```

## 训练：1D-ResNet-SE（多尺度卷积 + LayerNorm + Dropout）

```bash
python /root/毕设/Terra-main/model_train/new_models/train_classifier.py \
  --interim_root /root/毕设/Terra-main/people_database/interim_matlab \
  --dataset A3 \
  --train_subsets A3_1,A3_3 \
  --test_subset A3_2 \
  --model resnet1d_se \
  --dropout 0.25
```

## 训练：BiLSTM + 多头自注意力（mean+max pooling）

```bash
python /root/毕设/Terra-main/model_train/new_models/train_classifier.py \
  --interim_root /root/毕设/Terra-main/people_database/interim_matlab \
  --dataset A3 \
  --train_subsets A3_1,A3_3 \
  --test_subset A3_2 \
  --model bilstm_attn \
  --dropout 0.25
```

## 训练：融合模型（ResNet 1D + 多尺度 + SE + BiLSTM + 多头自注意力）

```bash
python /root/毕设/Terra-main/model_train/new_models/train_classifier.py \
  --interim_root /root/毕设/Terra-main/people_database/interim_matlab \
  --dataset A3 \
  --train_subsets A3_1,A3_3 \
  --test_subset A3_2 \
  --model hybrid \
  --hybrid_variant full \
  --dropout 0.25
```

## 消融（逐步添加模块）

固定同一划分（例如 A3_1,A3_3 → A3_2），只改变 `--hybrid_variant`：

- `base`：不加多尺度、不加SE、不加LSTM/注意力（仅1D-ResNet骨干）
- `ms`：在骨干前加入多尺度卷积（3/5/7）
- `ms_se`：在 `ms` 基础上加入 SE
- `ms_se_lstm`：在 `ms_se` 基础上加入 BiLSTM（无注意力）
- `full`：在 `ms_se_lstm` 基础上加入多头自注意力

示例：

```bash
python /root/毕设/Terra-main/model_train/new_models/train_classifier.py \
  --interim_root /root/毕设/Terra-main/people_database/interim_matlab \
  --dataset A3 \
  --train_subsets A3_1,A3_3 \
  --test_subset A3_2 \
  --model hybrid \
  --hybrid_variant ms_se_lstm \
  --dropout 0.25
```

训练结束会在 `model_train/results/new_models/<exp>/` 下生成：

- `best_model.pth`
- `training_history.png`（包含 Test Acc 横线）
- `metrics.json`
- `confusion_matrix_test.png`

## WGAN-GP（可选）：按子集（材质域）训练并生成合成样本

### 1) 训练 WGAN（对单个 subset）

```bash
python /root/毕设/Terra-main/model_train/new_models/train_wgan_gp.py \
  --interim_root /root/毕设/Terra-main/people_database/interim_matlab \
  --dataset A3 \
  --subset A3_1 \
  --epochs 50
```

### 2) 生成合成样本（保存为 .pt）

```bash
python /root/毕设/Terra-main/model_train/new_models/generate_synthetic.py \
  --wgan_dir /root/毕设/Terra-main/model_train/results/new_models_wgan/wgan_gp_A3_A3_1 \
  --out_path /root/毕设/Terra-main/model_train/results/new_models_wgan/syn_A3_1.pt \
  --num_per_class 200
```

### 3) 混入分类训练（synthetic_ratio 控制混入比例）

```bash
python /root/毕设/Terra-main/model_train/new_models/train_classifier.py \
  --interim_root /root/毕设/Terra-main/people_database/interim_matlab \
  --dataset A3 \
  --train_subsets A3_1,A3_3 \
  --test_subset A3_2 \
  --model resnet1d_se \
  --synthetic_pt /root/毕设/Terra-main/model_train/results/new_models_wgan/syn_A3_1.pt \
  --synthetic_ratio 0.5
```
