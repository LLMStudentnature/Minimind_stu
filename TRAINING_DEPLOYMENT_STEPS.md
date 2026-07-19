# MiniMind 环境要求与完整实现步骤

本文档只整理本项目从本地环境准备、训练到部署所需的环境与操作步骤。

## 1. 基础环境

### 1.1 推荐系统环境

- 操作系统：Ubuntu 20.04 或更新版本
- Python：3.10，项目 README 参考版本为 Python 3.10.16
- CUDA：推荐 CUDA 12.x，项目 README 参考版本为 CUDA 12.2
- PyTorch：按本机 CUDA 版本安装，建议先确认 GPU 可用
- Git：用于拉取代码和版本管理

检查 CUDA 是否可用：

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

如果输出 `True`，说明 PyTorch 能识别 CUDA GPU；如果输出 `False`，训练会走 CPU，速度会非常慢。

### 1.2 Python 依赖

在项目根目录执行：

```bash
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple
```

`requirements.txt` 中主要包括：

- `torch`：需要根据 CUDA 环境单独安装或确认版本
- `transformers`
- `datasets`
- `modelscope`
- `streamlit`
- `fastapi` / `uvicorn`
- `openai`
- `swanlab`
- `numpy`
- `scikit_learn`
- `sentence_transformers`

如果本机没有安装 PyTorch，建议先到 PyTorch 官方命令生成页选择对应 CUDA 版本安装，再安装项目依赖。

## 2. 物理环境要求

### 2.1 最低可运行环境

- CPU：普通多核 CPU 即可运行推理和小规模 LoRA
- 内存：建议 16 GB 以上，完整训练建议 32 GB 以上
- 显卡：没有 GPU 也能运行部分脚本，但训练速度非常慢
- 磁盘：建议至少预留 30 GB，完整数据集和多个权重会占用更多空间

### 2.2 推荐训练环境

项目 README 中作者参考环境：

- CPU：Intel i9-10980XE
- 内存：128 GB
- GPU：NVIDIA RTX 3090 24 GB，多卡环境为 8 张 3090
- 系统：Ubuntu 20.04
- CUDA：12.2
- Python：3.10.16

### 2.3 各训练阶段建议硬件

| 阶段 | 是否必须 | 默认数据 | 默认前置权重 | 推荐物理环境 | 输出权重 |
| --- | --- | --- | --- | --- | --- |
| 预训练 Pretrain | 必须 | `dataset/pretrain_t2t_mini.jsonl` | 无，从 0 开始 | 单张 24 GB GPU 推荐；CPU 可跑但非常慢 | `out/pretrain_768.pth` |
| 全参数 SFT | 必须 | `dataset/sft_t2t_mini.jsonl` | `pretrain` | 单张 24 GB GPU 推荐 | `out/full_sft_768.pth` |
| LoRA 微调 | 可选 | 自定义 `lora_xxx.jsonl` | `full_sft` | CPU 或单张 GPU 均可，GPU 更快 | `out/lora_xxx_768.pth` |
| DPO | 可选 | `dataset/dpo.jsonl` | `full_sft` | 单张 24 GB GPU 推荐 | `out/dpo_768.pth` |
| PPO | 可选 | `dataset/rlaif.jsonl` | `full_sft` | 单张 24 GB GPU 起步，更推荐多卡；还需要 Reward Model | `out/ppo_actor_768.pth` |
| GRPO / CISPO | 可选 | `dataset/rlaif.jsonl` | `full_sft` | 单张 24 GB GPU 起步，更推荐多卡；还需要 Reward Model | `out/grpo_768.pth` |
| Agentic RL | 可选 | `dataset/agent_rl.jsonl` | `full_sft` | 单张 24 GB GPU 起步，更推荐多卡；可选 SGLang 加速 rollout | `out/agent_768.pth` |
| 蒸馏 Distillation | 可选 | `dataset/sft_t2t_mini.jsonl` | 学生和教师各自 `full_sft` | 需要同时加载教师和学生模型，显存要求高于普通 SFT | `out/full_dist_768.pth` |

### 2.4 训练耗时参考

基于单张 NVIDIA RTX 3090，使用 mini 数据集训练 `minimind-3`：

- `pretrain_t2t_mini`：约 1.21 小时
- `sft_t2t_mini`：约 1.10 小时
- Pretrain + SFT 从 0 训练出基础对话模型：约 2.31 小时

`minimind-3-moe` 更慢，Pretrain + SFT 约 3.23 小时。

以上只是参考值，实际耗时取决于 GPU、batch size、数据长度、磁盘速度和是否使用多卡。

## 3. 数据准备

### 3.1 下载数据集

从 ModelScope 或 HuggingFace 下载 MiniMind 数据集文件，然后放到项目的 `dataset/` 目录。

推荐快速复现只需要：

```text
dataset/pretrain_t2t_mini.jsonl
dataset/sft_t2t_mini.jsonl
```

完整或可选训练还会用到：

```text
dataset/pretrain_t2t.jsonl
dataset/sft_t2t.jsonl
dataset/dpo.jsonl
dataset/rlaif.jsonl
dataset/agent_rl.jsonl
dataset/agent_rl_math.jsonl
dataset/lora_xxx.jsonl
```

### 3.2 推荐数据组合

快速从 0 训练基础对话模型：

```text
pretrain_t2t_mini.jsonl + sft_t2t_mini.jsonl
```

完整复现主线模型：

```text
pretrain_t2t.jsonl + sft_t2t.jsonl + rlaif.jsonl / agent_rl.jsonl
```

## 4. 完整训练步骤

所有训练脚本建议进入 `trainer/` 目录执行。

### 4.1 预训练

单卡训练：

```bash
cd trainer
python train_pretrain.py
```

多卡训练，把 `N` 改成 GPU 数量：

```bash
cd trainer
torchrun --nproc_per_node N train_pretrain.py
```

默认输出：

```text
out/pretrain_768.pth
```

如果中断后续训：

```bash
python train_pretrain.py --from_resume 1
```

常用参数示例：

```bash
python train_pretrain.py \
  --data_path ../dataset/pretrain_t2t_mini.jsonl \
  --epochs 2 \
  --batch_size 32 \
  --max_seq_len 340 \
  --hidden_size 768 \
  --num_hidden_layers 8
```

### 4.2 全参数 SFT

SFT 默认从 `pretrain` 权重继续训练，所以要先完成预训练，或者提前把已有 `pretrain_768.pth` 放到 `out/` 目录。

```bash
cd trainer
python train_full_sft.py
```

多卡训练：

```bash
torchrun --nproc_per_node N train_full_sft.py
```

默认输出：

```text
out/full_sft_768.pth
```

如果中断后续训：

```bash
python train_full_sft.py --from_resume 1
```

常用参数示例：

```bash
python train_full_sft.py \
  --data_path ../dataset/sft_t2t_mini.jsonl \
  --from_weight pretrain \
  --epochs 2 \
  --batch_size 16 \
  --max_seq_len 768 \
  --hidden_size 768 \
  --num_hidden_layers 8
```

### 4.3 测试训练后的模型

回到项目根目录：

```bash
cd ..
python eval_llm.py --weight full_sft
```

测试预训练权重：

```bash
python eval_llm.py --weight pretrain
```

开启思考输出：

```bash
python eval_llm.py --weight full_sft --open_thinking 1
```

如果是 MoE 权重：

```bash
python eval_llm.py --weight full_sft --use_moe 1
```

## 5. 可选训练步骤

### 5.1 LoRA 微调

准备自定义多轮对话格式数据，例如：

```text
dataset/lora_medical.jsonl
```

执行：

```bash
cd trainer
python train_lora.py \
  --data_path ../dataset/lora_medical.jsonl \
  --lora_name lora_medical \
  --from_weight full_sft
```

默认输出：

```text
out/lora_medical_768.pth
```

测试 LoRA：

```bash
cd ..
python eval_llm.py --weight full_sft --lora_weight lora_medical
```

### 5.2 DPO 偏好优化

需要：

```text
dataset/dpo.jsonl
out/full_sft_768.pth
```

执行：

```bash
cd trainer
python train_dpo.py
```

默认输出：

```text
out/dpo_768.pth
```

测试：

```bash
cd ..
python eval_llm.py --weight dpo
```

### 5.3 PPO 强化学习

需要：

```text
dataset/rlaif.jsonl
out/full_sft_768.pth
Reward Model，例如 ../../internlm2-1_8b-reward
```

执行：

```bash
cd trainer
python train_ppo.py \
  --data_path ../dataset/rlaif.jsonl \
  --from_weight full_sft \
  --reward_model_path ../../internlm2-1_8b-reward
```

默认输出：

```text
out/ppo_actor_768.pth
```

### 5.4 GRPO / CISPO 强化学习

`train_grpo.py` 默认 `loss_type=cispo`，如需 GRPO 可显式传入 `--loss_type grpo`。

需要：

```text
dataset/rlaif.jsonl
out/full_sft_768.pth
Reward Model，例如 ../../internlm2-1_8b-reward
```

执行 CISPO：

```bash
cd trainer
python train_grpo.py \
  --data_path ../dataset/rlaif.jsonl \
  --from_weight full_sft \
  --loss_type cispo \
  --reward_model_path ../../internlm2-1_8b-reward
```

执行 GRPO：

```bash
python train_grpo.py \
  --data_path ../dataset/rlaif.jsonl \
  --from_weight full_sft \
  --loss_type grpo \
  --reward_model_path ../../internlm2-1_8b-reward
```

默认输出：

```text
out/grpo_768.pth
```

### 5.5 Agentic RL

需要：

```text
dataset/agent_rl.jsonl
out/full_sft_768.pth
Reward Model，例如 ../../internlm2-1_8b-reward
```

执行：

```bash
cd trainer
python train_agent.py \
  --data_path ../dataset/agent_rl.jsonl \
  --from_weight full_sft \
  --reward_model_path ../../internlm2-1_8b-reward
```

使用数学 Agent 数据：

```bash
python train_agent.py \
  --data_path ../dataset/agent_rl_math.jsonl \
  --from_weight full_sft \
  --reward_model_path ../../internlm2-1_8b-reward
```

默认输出：

```text
out/agent_768.pth
```

### 5.6 蒸馏训练

需要同时准备学生模型和教师模型权重。默认学生是 Dense，教师是 MoE。

```bash
cd trainer
python train_distillation.py \
  --data_path ../dataset/sft_t2t_mini.jsonl \
  --from_student_weight full_sft \
  --from_teacher_weight full_sft
```

默认输出：

```text
out/full_dist_768.pth
```

## 6. 部署环境

### 6.1 本地部署最低环境

- Python 3.10
- 已安装 `requirements.txt`
- 已有可推理模型权重
- CPU 可运行，但速度慢
- 推荐使用 NVIDIA GPU，显存 8 GB 以上更合适

### 6.2 服务部署推荐环境

- Linux 服务器
- NVIDIA GPU
- CUDA 和 PyTorch GPU 版本正常
- 开放服务端口，例如 `8998`
- 如需公网访问，配置安全组、防火墙、反向代理或内网穿透

## 7. 完整部署步骤

### 7.1 CLI 本地推理

使用 PyTorch 权重：

```bash
python eval_llm.py --load_from ./model --weight full_sft
```

使用 Transformers 格式模型：

```bash
python eval_llm.py --load_from ./minimind-3
```

### 7.2 Streamlit WebUI

如果使用 Transformers 格式模型，需要先把模型目录放到 `scripts/` 目录下，例如：

```bash
cp -r minimind-3 scripts/minimind-3
```

启动 WebUI：

```bash
cd scripts
streamlit run web_demo.py
```

浏览器访问 Streamlit 输出的本地地址，一般是：

```text
http://localhost:8501
```

### 7.3 OpenAI-compatible API 服务

启动服务：

```bash
cd scripts
python serve_openai_api.py
```

默认监听：

```text
http://localhost:8998
```

默认加载：

```text
../model tokenizer
../out/full_sft_768.pth 权重
```

指定权重示例：

```bash
python serve_openai_api.py --weight full_sft --hidden_size 768 --num_hidden_layers 8
```

指定 Transformers 格式模型目录：

```bash
python serve_openai_api.py --load_from ../minimind-3
```

测试接口：

```bash
python chat_api.py
```

或者用 `curl`：

```bash
curl http://localhost:8998/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "minimind",
    "messages": [
      {"role": "user", "content": "世界上最高的山是什么？"}
    ],
    "temperature": 0.7,
    "max_tokens": 1024,
    "stream": true,
    "open_thinking": true
  }'
```

### 7.4 vLLM 部署

需要 Transformers 格式模型目录，并且需要 CUDA 环境。

```bash
vllm serve /path/to/model \
  --model-impl transformers \
  --served-model-name minimind \
  --port 8998
```

### 7.5 Ollama 部署

如果直接使用官方发布模型：

```bash
ollama run jingyaogong/minimind-3
```

## 8. 推荐完整路线

### 8.1 快速从 0 复现基础模型

```bash
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple

# 下载并放置数据
# dataset/pretrain_t2t_mini.jsonl
# dataset/sft_t2t_mini.jsonl

cd trainer
python train_pretrain.py
python train_full_sft.py

cd ..
python eval_llm.py --weight full_sft
```

### 8.2 训练后部署 API

```bash
cd scripts
python serve_openai_api.py --weight full_sft
```

访问：

```text
http://localhost:8998/v1/chat/completions
```

### 8.3 训练后部署 WebUI

```bash
cd scripts
streamlit run web_demo.py
```

访问：

```text
http://localhost:8501
```

## 9. 常见注意事项

- `train_full_sft.py` 默认依赖 `out/pretrain_768.pth`。
- 所有训练输出默认保存在 `out/` 目录。
- 断点续训文件默认保存在 `checkpoints/` 目录。
- 多卡训练使用 `torchrun --nproc_per_node N train_xxx.py`。
- `--hidden_size`、`--num_hidden_layers`、`--use_moe` 必须和权重结构保持一致。
- MoE 权重文件名会带 `_moe` 后缀。
- RL 阶段通常需要额外 Reward Model，默认路径是 `../../internlm2-1_8b-reward`。
- CPU 可以运行推理和部分小训练，但完整训练建议使用 NVIDIA GPU。
- 部署到服务器时需要确认端口 `8998` 或 `8501` 是否开放。
