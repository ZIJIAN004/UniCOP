# UniCOP-Reason · OpenRLHF 实验分支

## 存在理由

父目录当前用 **TRL 1.1.0 + vLLM-serve** 跑 GRPO。踩到多处版本兼容坑（FlashInfer 版本锁、vLLM V1 logits processor API 变动、LoRA adapter 模块 bug），维护成本上升。

本子目录在**不动父目录任何代码**的前提下，平行实现一套 **OpenRLHF** 版本的 GRPO 训练流程。两者共享：

- SFT 初始权重（`/Data04/yangzhihan/lzj/UniCOP-Distill/output_sft_r1_v2/merged_model`）
- 数据集定义（父目录 `problems/`）
- POMO PRM 和 terminal reward 算法（父目录 `pomo_prm.py` / `terminal_reward.py`）

但**完全隔离**：

- Python 环境：`/Data04/yangzhihan/envs/openrlhf_env`（独立 conda env）
- 输出 ckpt：`openrlhf/output/`（不写父目录 `output/`）
- 训练脚本、reward wrapper、logits processor：全部新写

## 与父目录 TRL 版本的能力对比

| 能力 | 父目录 TRL | 本目录 OpenRLHF |
|---|---|---|
| GRPO 算法 | ✅ | ✅ |
| DAPO clip-higher（非对称 ratio） | ✅ 手写 | ✅ 原生 `--algo dapo` |
| POMO PRM（step-level） | ✅ 走 `grpo_prm_trainer.py` 改装 | ⚠️ 聚合为 scalar（远程 reward 接口限制） |
| Terminal reward | ✅ | ✅ |
| NoRepeatNgram（n=6） | ✅ 走 `trl vllm-serve --logits-processors` | ✅ 本目录 `custom/ngram_processor.py` |
| LoRA rank=64 | ✅ | ✅ |
| ZeRO-3 / FSDP | ✅ ZeRO-3 | ✅ FSDP2（推荐）或 ZeRO-3 |
| vLLM rollout | ✅ 独立 1 卡 | ✅ Ray 托管 |
| 调度自动化 | ⚠️ 手写 200 行 bash | ✅ Ray 原生 |

### 已知局限：step-level PRM

OpenRLHF 的 `--remote_rm_url` 接口返回**completion 级 scalar reward**，无法像父目录 `grpo_prm_trainer.py` 那样在 token 位置提供 per-step advantage。

当前策略：**将 PRM 逐步信号按客户节点数归一化后求和，作为完整 completion 的聚合分量**，与 terminal reward 相加返回。

这是**妥协方案**，信号密度不如父目录实现。如果发现收敛变慢或偏弱，下一步要 fork OpenRLHF 的 `experience_maker.py`——详见 `reward/remote_reward_server.py` 内 `# TODO: step-level` 注释块。

## 目录结构

```
openrlhf/
├── README.md                  本文件
├── requirements.txt           OpenRLHF 版依赖锁
├── install.sh                 conda env 创建 + 完整安装（含 flash-attn wheel 修复）
├── scripts/
│   ├── verify_env.py          环境自检（所有包版本、CUDA、FlashAttn 功能）
│   └── smoke_test_vllm.py     vLLM 烟雾测试
├── data/
│   └── prepare_dataset.py     problems/ → OpenRLHF jsonl
├── reward/
│   ├── __init__.py
│   ├── _shared.py             通过 sys.path 复用父目录 POMO/terminal 代码
│   ├── remote_reward_server.py  FastAPI 远程 reward 服务（训练时起一个进程）
│   └── reward_fn.py           本地模式 reward（colocate 训练用，不走 HTTP）
├── custom/
│   ├── __init__.py
│   └── ngram_processor.py     vLLM V1 NoRepeatNgram logits processor
├── configs/
│   └── train_grpo_tsp10_1.5b.sh  启动脚本（8 卡 3090 · 1.5B · LoRA · DAPO）
└── output/                    ckpt 输出，git-ignore
```

## 跑通流程（首次）

```bash
# 1. 装环境（只做一次）
cd /Data04/yangzhihan/lzj/UniCOP-Reason/openrlhf
bash install.sh

# 2. 自检
conda activate /Data04/yangzhihan/envs/openrlhf_env
python scripts/verify_env.py
python scripts/smoke_test_vllm.py \
  --model /Data04/yangzhihan/lzj/UniCOP-Distill/output_sft_r1_v2/merged_model

# 3. 准备数据（只做一次，产物写到 data/processed/）
python data/prepare_dataset.py --problem_type tsp --problem_size 10 --num_train 20000

# 4. 起远程 reward server（一个终端常驻）
python reward/remote_reward_server.py --problem_type tsp --problem_size 10 --port 5000

# 5. 另一个终端启动训练
bash configs/train_grpo_tsp10_1.5b.sh
```

## 迁移进度

- [x] 环境安装 + 自检脚本（`scripts/`）
- [x] 数据准备（`data/prepare_dataset.py`）
- [x] Reward server（`reward/remote_reward_server.py`）
- [x] NoRepeatNgram processor（`custom/ngram_processor.py`）
- [x] 启动脚本（`configs/train_grpo_tsp10_1.5b.sh`）
- [ ] 与父目录 TRL 版本跑同一组实验对比收敛曲线
- [ ] 若聚合 PRM 信号不够，fork experience_maker 实现真·step-level
