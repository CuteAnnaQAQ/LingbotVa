
## 1. 基础运行参数

```python
va_robomme_cfg = EasyDict(__name__='Config: VA RoboMME')
va_robomme_cfg.update(va_shared_cfg)
va_robomme_cfg.infer_mode = 'server'
```


| 参数                    | 作用                                  | 是否需要确定 | 在哪里确定                                                                                                |
| ------------------------- | --------------------------------------- | -------------: | ----------------------------------------------------------------------------------------------------------- |
| `__name__`              | 配置名称，仅用于标识                  |           否 | 保持当前值                                                                                                |
| `update(va_shared_cfg)` | 继承端口、精度、patch size 等公共配置 |           否 | [shared_config.py](E:/Science/Embodied_AI/Lingbotva_Robomme/lingbot-va/wan_va/configs/shared_config.py:1) |
| `infer_mode='server'`   | 运行 WebSocket 推理服务器             |           否 | RoboMME 使用远程 policy adapter，因此应为`server`                                                         |

## 2. Checkpoint 路径

```python
va_robomme_cfg.wan22_pretrained_model_name_or_path = (
    "/DATA/disk0/yry/lingbot-va-posttrain"
)
```

服务器会从这个目录加载：

```text
lingbot-va-posttrain/
├── vae/
├── tokenizer/
├── text_encoder/
└── transformer/
```

加载逻辑在 [wan_va_server.py](E:/Science/Embodied_AI/Lingbotva_Robomme/lingbot-va/wan_va/wan_va_server.py:51)。

这是必须确认的参数：

- 目录必须真实存在于运行 LingBot 服务的 Linux 机器上。
- 必须是 RoboMME post-train 后的 checkpoint。
- 训练时必须使用相同的：
  - action mapping
  - action 表示方式
  - normalization statistics
  - 相机顺序
  - 时间采样方式

如果该目录实际是 Libero checkpoint，仅修改 inference config 不会使它学会 Panda joint control，只能用于接口联调。

## 3. 时间与缓存参数

```python
va_robomme_cfg.attn_window = 30
va_robomme_cfg.frame_chunk_size = 4
va_robomme_cfg.action_per_frame = 4
```

### `frame_chunk_size`

表示一次 inference chunk 包含多少个 VAE latent 时间帧。

当前：

```text
frame_chunk_size = 4
```

服务器一次产生：

```text
4 个 latent frames
4 × action_per_frame 个机器人动作
```

训练代码会随机使用 `chunk_size=1...4`，见 [train.py](E:/Science/Embodied_AI/Lingbotva_Robomme/lingbot-va/wan_va/train.py:245)，所以对当前训练实现而言，推理设置为 `4` 是合理的。

状态：基本可以确定为 `4`，除非 RoboMME post-train 时改过训练代码。

### `action_per_frame`

这是目前最需要确定的参数。

它表示每个 VAE latent frame 对应多少个底层 RoboMME control action。数据加载器的实际计算关系是：

```python
frame_stride = latent_frame_ids[1] - latent_frame_ids[0]
action_per_frame = frame_stride * 4
```

原因是 Wan VAE 时间方向大约每 4 个采样视频帧形成一个 latent frame。对应实现见 [lerobot_latent_dataset.py](E:/Science/Embodied_AI/Lingbotva_Robomme/lingbot-va/wan_va/dataset/lerobot_latent_dataset.py:256)。

例如：


| RoboMME 原始 FPS | latent 提取 FPS | `frame_stride` | `action_per_frame` |
| -----------------: | ----------------: | ---------------: | -------------------: |
|               30 |              30 |              1 |                  4 |
|               30 |              15 |              2 |                  8 |
|               30 |              10 |              3 |                 12 |
|               20 |              10 |              2 |                  8 |

当前：

```python
action_per_frame = 4
```

只在 `frame_stride=1` 时正确。

应当去生成的 latent 文件中确定。每个 `.pth` 中已有：

```text
frame_ids
fps
ori_fps
```

检查方式类似：

```python
data = torch.load("episode_xxx.pth", map_location="cpu")
print(data["frame_ids"][:10])
print(data["fps"], data["ori_fps"])
```

然后计算：

```python
frame_stride = data["frame_ids"][1] - data["frame_ids"][0]
action_per_frame = frame_stride * 4
```

该值必须和训练时一致。

### `attn_window`

这是流式 KV cache 保存的历史窗口大小。

代码把缓存平均分给 video token 和 action token：

```python
(attn_window // 2) * latent_token_per_chunk
(attn_window // 2) * action_token_per_chunk
```

见 [model.py](E:/Science/Embodied_AI/Lingbotva_Robomme/lingbot-va/wan_va/modules/model.py:661)。

当前：

```text
attn_window = 30
```

相当于最多为两种模态分别预留约 15 个 chunk。若：

```text
frame_chunk_size = 4
```

则 video 一侧约能容纳：

```text
15 × 4 = 60 个 latent frames
```

这是可调参数：

- 越大：能保留更长历史，但显存消耗更高。
- 越小：显存更少，但 RoboMME memory task 容易丢失历史信息。
- 当前训练代码的 `window_size` 随机范围是 `4...64`，因此建议保持在该范围内。
- `30` 可以先用于联调，正式 memory benchmark 再根据显存和历史长度调大。

建议使用偶数，因为实现通过 `attn_window // 2` 平分 video/action cache。

## 4. 环境和图像参数

```python
va_robomme_cfg.env_type = 'none'

va_robomme_cfg.height = 256
va_robomme_cfg.width = 256

va_robomme_cfg.obs_cam_keys = [
    'image',
    'wrist_image',
]
```

### `env_type`

`robotwin_tshape` 会走 RoboTwin 特有的三相机拼接和双 VAE 路径。

RoboMME 不需要这个特殊逻辑，因此：

```python
env_type = 'none'
```

是确定值。

### `height/width`

服务器会将每路输入图像 resize 到这里指定的尺寸，见 [wan_va_server.py](E:/Science/Embodied_AI/Lingbotva_Robomme/lingbot-va/wan_va/wan_va_server.py:332)。

RoboMME 的 base camera 原生为 256×256，因此当前设置合理：

```python
height = 256
width = 256
```

但最终应当以 post-train 时 latent 提取的尺寸为准。训练数据与推理最好完全一致。

尺寸还应满足模型空间压缩和 patch 划分，建议使用 32 的整数倍。

### `obs_cam_keys`

这两个字符串同时对齐 RoboMME 官方 LeRobot 数据集的 feature key 和 LingBot WebSocket observation dict 的键名。

RoboMME 原始输出通常是：

```python
obs["front_rgb_list"]
obs["wrist_rgb_list"]
```

RoboMME policy adapter 应转换为：

```python
lingbot_obs = {
    "image": front_rgb,
    "wrist_image": wrist_rgb,
}
```

必须确认：

- LeRobot 训练数据使用同样的 key。
- latent 目录也使用同样的 key。
- inference adapter 发送相同的 key。
- 相机顺序完全相同。

非 RoboTwin 模式下，LingBot 会按列表顺序横向拼接 latent，因此顺序不能交换。

## 5. Action 空间参数

```python
va_robomme_cfg.action_dim = 30
```

这是 LingBot canonical action head 的维度，不是 RoboMME 环境动作维度。

必须保持：

```text
action_dim = 30
```

不能改成 8，否则 transformer action embedding/output head 的形状和 checkpoint 不匹配。

RoboMME 对外仍然收到 8 维：

```text
[j0, j1, j2, j3, j4, j5, j6, gripper]
```

## 6. Action channel mapping

```python
va_robomme_cfg.used_action_channel_ids = (
    list(range(14, 21)) + [28]
)
```

这是已经确定的接口：


| RoboMME   |   LingBot |
| ----------- | ----------: |
| `j0...j6` | `14...20` |
| `gripper` |      `28` |

不需要调整。

### `inverse_used_action_channel_ids`

它把 RoboMME 8D action 展开为 LingBot 30D：

```text
RoboMME 8D → LingBot canonical 30D
```

未使用的 channel 全部指向额外补出的第 9 个零值。随后模型还会使用 `action_mask` 把未使用的 22 个 channel 清零，见 [wan_va_server.py](E:/Science/Embodied_AI/Lingbotva_Robomme/lingbot-va/wan_va/wan_va_server.py:413)。

这是自动生成值，不应手工填写。

## 7. Action normalization

```python
va_robomme_cfg.action_norm_method = 'quantiles'
```

归一化公式为：

```text
normalized =
    (action - q01) / (q99 - q01 + 1e-6) * 2 - 1
```

反归一化为逆过程，见 [wan_va_server.py](E:/Science/Embodied_AI/Lingbotva_Robomme/lingbot-va/wan_va/wan_va_server.py:225)。

`quantiles` 是当前训练管线使用的方法，可保持不变。

### 当前 `q01/q99`

当前代码中内置的关节范围来自 RoboMME train-set 的 state quantile，只用于无数据时的接口 smoke test，不是正式 action 统计量：

```python
robomme_action_q01 = [...]
robomme_action_q99 = [...]
```

正式训练和评测默认使用官方 LeRobot 原始的 `absolute_joint` 8D action。先计算精确统计量：

```bash
python -m wan_va.utils.compute_robomme_action_stats \
  /path/to/robomme_data_lerobot \
  --output /path/to/robomme_action_stats.json
```

再使训练和推理都加载同一文件：

```bash
export LINGBOT_ROBOMME_ACTION_STATS=/path/to/robomme_action_stats.json
```

正确流程是：

1. LingBot RoboMME 默认 action 为 official LeRobot `absolute_joint` target。
2. 服务端反归一化后返回 `action_representation` 和 `conditioned_action_frames`，客户端据此完成最终转换。
3. 对 RoboMME train split 的最终 8D action 计算每一维的 1% 和 99% 分位数。
4. 将 8D 统计量映射到 channel `14...20, 28`。
5. 训练和推理使用完全相同的统计量。

特别注意：仓库 [norm_stats.json](E:/Science/Embodied_AI/Lingbotva_Robomme/robomme_policy_learning/assets/norm_stats.json:1) 中的官方 `actions.q01/q99`，前 7 维是经过 `DeltaActions` 转换后的统计量。相关转换在 [training/config.py](E:/Science/Embodied_AI/Lingbotva_Robomme/robomme_policy_learning/src/mme_vla_suite/training/config.py:370)。

如果 LingBot 直接预测 absolute joint target，就不能直接复制那组 delta-action 统计量。

如果 `/DATA/disk0/yry/lingbot-va-posttrain` 已经训练完成，应优先找到当时保存的训练配置或统计文件，直接使用训练时的值，不能重新计算出另一套。

## 8. CFG 参数

```python
va_robomme_cfg.guidance_scale = 5
va_robomme_cfg.action_guidance_scale = 1
```

### `guidance_scale`

控制视频生成的文本 classifier-free guidance：

- `1`：关闭 CFG。
- `>1`：启用 CFG。
- 越大：生成视频更服从语言，但显存和计算量增加，过大可能失真。

当前 `5` 沿用 LingBot 默认设置，可先保留。

### `action_guidance_scale`

控制 action diffusion 的 CFG：

```python
action_guidance_scale = 1
```

表示 action 不做额外 CFG。

这是推理可调参数，不要求严格等于训练参数，但只有训练时做过 unconditional prompt dropout，CFG 才有意义。训练配置中的对应基础是：

```python
cfg_prob = 0.1
```

当前建议保持 `1`，等基本成功率跑通后再做 `1/2/3/5` 消融。

启用任意 CFG 都会使用双 batch，增加显存和推理时间。

## 9. Diffusion 推理步数

```python
va_robomme_cfg.num_inference_steps = 20
va_robomme_cfg.action_num_inference_steps = 50
va_robomme_cfg.video_exec_step = -1
```

### `num_inference_steps`

视频 latent 的去噪步数：

- 越大：视频预测通常更稳定。
- 越大：推理越慢。
- 视频预测又会进入 action 推理的历史上下文，因此不只是“可视化质量”。

当前 `20` 可作为初始值。

### `action_num_inference_steps`

action diffusion 去噪步数：

- 直接影响动作质量和延迟。
- 当前 `50` 偏向质量。
- 接口联调可先降到 `10`。
- 正式评测建议从 `50` 开始，再测试 `10/20/30/50` 的成功率与延迟。

这些是推理超参数，不需要与训练步数完全一致。

### `video_exec_step`

```python
video_exec_step = -1
```

表示完整执行视频去噪流程并在最终 clean pass 更新 cache。

如果设置成正整数，会提前截断视频生成循环，可以提速，但会改变写入 KV cache 的视频状态，可能影响 action 质量。

建议正式评测保持 `-1`。

## 10. Flow-matching scheduler 参数

```python
va_robomme_cfg.snr_shift = 5.0
va_robomme_cfg.action_snr_shift = 0.05
```

分别控制：

- `snr_shift`：视频 diffusion 的噪声时间表。
- `action_snr_shift`：action diffusion 的噪声时间表。

实现位置：

- inference：[wan_va_server.py](E:/Science/Embodied_AI/Lingbotva_Robomme/lingbot-va/wan_va/wan_va_server.py:51)
- training：[train.py](E:/Science/Embodied_AI/Lingbotva_Robomme/lingbot-va/wan_va/train.py:138)

这两个参数会同时影响训练噪声分布和推理去噪轨迹，因此应当与 RoboMME post-train 时使用的配置一致，不能只在 inference config 中随意改。

当前 `5.0/0.05` 是从 Libero 配置继承的。如果 RoboMME post-train 也使用这份配置，则保持；否则去训练日志或训练 config 中找真实值。

## 11. 公共参数

来自 [shared_config.py](E:/Science/Embodied_AI/Lingbotva_Robomme/lingbot-va/wan_va/configs/shared_config.py:1)。

### `host`

```python
host = '0.0.0.0'
```

监听所有网卡，远程 RoboMME adapter 可以连接。服务器在公网环境时需要防火墙限制。

### `port`

```python
port = 29536
```

必须与 RoboMME client 使用的端口一致，并确保未被占用。

启动时也可以覆盖：

```bash
CONFIG_NAME=robomme NGPU=1 \
bash script/run_launch_va_server_sync.sh --port 29536
```

注意脚本中虽然定义了 `PORT` 环境变量，但当前没有直接把它传给 Python；使用 `--port` 更明确。

### `param_dtype`

```python
param_dtype = torch.bfloat16
```

需要 GPU 支持 BF16。A100、H100、4090 等一般可以；较老 GPU 可能需要 `float16`。

最好与 checkpoint 训练/保存精度一致。

### `patch_size`

```python
patch_size = (1, 2, 2)
```

这是 transformer 架构参数，必须与 checkpoint 一致，不应调整。

### `enable_offload`

```python
enable_offload = False
```

- `False`：VAE、text encoder 常驻 GPU，速度更快、显存更多。
- `True`：部分模型留在 CPU，需要时搬到 GPU，节省显存、降低速度。

只根据实际显存决定，不影响动作接口。

### `save_root`

```python
save_root = './train_out'
```

保存调试用 latents、actions 和 observation。长时间 benchmark 可能产生较多文件，要确认磁盘空间。

## 当前参数状态汇总


| 参数                                  | 当前状态                            |
| --------------------------------------- | ------------------------------------- |
| `action_dim=30`                       | 已确定                              |
| `used_action_channel_ids=[14..20,28]` | 已确定                              |
| `inverse_used_action_channel_ids`     | 自动生成                            |
| `env_type='none'`                     | 已确定                              |
| `infer_mode='server'`                 | 已确定                              |
| `height/width=256`                    | 基本确定，但需和训练 latent 核对    |
| `obs_cam_keys`                        | 需和数据转换及 client adapter 核对  |
| `frame_chunk_size=4`                  | 当前训练代码支持，基本确定          |
| `action_per_frame=4`                  | 必须根据 latent`frame_ids` 重新确认 |
| `attn_window=30`                      | 可调，正式 memory 评测前确定        |
| `norm_stat`                           | 当前只是 bootstrap，必须替换/核对   |
| `snr_shift/action_snr_shift`          | 必须和 post-train 配置一致          |
| `guidance_scale`                      | 推理验证集调优                      |
| `num_inference_steps`                 | 延迟/成功率调优                     |
| checkpoint 路径                       | 必须确认 checkpoint 内容及训练配置  |
