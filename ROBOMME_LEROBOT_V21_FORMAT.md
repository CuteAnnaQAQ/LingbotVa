# RoboMME LeRobot v2.1 与 LingBot 数据格式

## 结论

`Yinpei/robomme_data_lerobot` 的原始数据可以被 `lerobot==0.3.3` 读取：
LeRobot 0.3.3 的代码格式版本正是 v2.1，官方数据也声明
`codebase_version: "v2.1"`。

但原始数据不能直接用于 LingBot 后训练。LingBot 训练 loader 不直接读取 RGB，
而是读取预先生成的 Wan2.2 VAE latent；同时还要求在 `episodes.jsonl` 中加入
`action_config`，并准备 `empty_emb.pt` 和与当前 action representation 一致的
8D normalization statistics。

## 兼容性矩阵


| 部分             | RoboMME 官方数据                            | LeRobot 0.3.3       | LingBot 训练                                        | 结论                  |
| ------------------ | --------------------------------------------- | --------------------- | ----------------------------------------------------- | ----------------------- |
| `meta/info.json` | `codebase_version=v2.1`                     | 原生版本            | loader 固定`revision=v2.1`                          | 兼容                  |
| front RGB        | `image`, `dtype=image`, `[256,256,3]`       | 从 Parquet 解码     | 需生成`latents/.../image/*.pth`                     | 原始兼容，需预处理    |
| wrist RGB        | `wrist_image`, `dtype=image`, `[256,256,3]` | 从 Parquet 解码     | 需生成`latents/.../wrist_image/*.pth`               | 原始兼容，需预处理    |
| action           | `actions`, `float32[8]`                     | 可直接读取          | 配置已映射到 30D 规范通道                           | 兼容                  |
| language         | `episodes.jsonl.tasks`                      | 可读取              | 需映射到`action_config[].action_text`/`text_emb`    | 需补充分段            |
| episode          | metadata 与 Parquet 都有`episode_index`     | 可索引              | 用于定位 Parquet 和 latent                          | 兼容                  |
| frame time/index | Parquet 有`timestamp`, `frame_index`        | 可读取并校验        | latent 用`frame_ids` 对齐 action                    | 兼容，latent 必须同源 |
| `videos/`        | `total_videos=0`                            | 不需要 MP4          | latent extractor 必须能读 Parquet RGB，或先导出视频 | 不属于原始数据缺失    |
| `action_config`  | 不存在                                      | 非 LeRobot 必需字段 | 必需                                                | 不兼容，需生成        |
| `latents/`       | 不存在                                      | 非 LeRobot 字段     | 必需                                                | 不兼容，需离线提取    |

## RoboMME 原始目录

```text
robomme_data_lerobot/
├── meta/
│   ├── info.json
│   ├── episodes.jsonl
│   ├── episodes_stats.jsonl
│   └── tasks.jsonl
└── data/
    ├── chunk-000/
    │   ├── episode_000000.parquet
    │   └── ...
    └── chunk-001/
        └── ...
```

官方 `info.json.video_path` 虽然保留了路径模板，但两路相机 feature 的 dtype 都是
`image`，且 `total_videos=0`。LeRobot 仅对 dtype 为 `video` 的 feature 访问 MP4；
本数据的 RGB 图片字节存储在每个 episode 的 Parquet 中，因此 `videos/` 可以没有。

LeRobot v2.1 不只需要用户点名的两个 metadata 文件。LingBot 使用的
`LeRobotDatasetMetadata` 还会读取 `meta/tasks.jsonl` 和
`meta/episodes_stats.jsonl`，这两个文件在官方发布中均存在。

## 每帧字段映射


| 语义                | RoboMME 字段            | shape/type         | LingBot 使用方式                                |
| --------------------- | ------------------------- | -------------------- | ------------------------------------------------- |
| front RGB           | `image`                 | image`[256,256,3]` | VAE latent camera key`image`                    |
| wrist RGB           | `wrist_image`           | image`[256,256,3]` | VAE latent camera key`wrist_image`              |
| robot state         | `state`                 | `float32[8]`       | delta 模式可选；absolute 模式不参与 action 转换 |
| 8D joint action     | `actions`               | `float32[8]`       | `[panda_j0..panda_j6, gripper]`                 |
| timestamp           | `timestamp`             | `float32`          | 应等于`frame_index / fps`，官方 fps 为 10       |
| episode-local index | `frame_index`           | `int64`            | 应从 0 连续递增                                 |
| episode index       | `episode_index`         | `int64`            | 应与所属 Parquet episode 一致                   |
| global row index    | `index`                 | `int64`            | LeRobot 额外索引，LingBot 当前不直接使用        |
| language mapping    | `task_index` + metadata | `int64`            | 原始语言取`episodes.jsonl.tasks`                |

RoboMME 的 8D action 被放到 LingBot 30D canonical action 的以下通道：

```text
RoboMME actions[0:7] -> LingBot channels[14:21]
RoboMME actions[7]   -> LingBot channel[28]
其余 canonical channels 填 0，并通过 mask 忽略
```

当前 RoboMME 配置使用 `absolute_joint`：Parquet 中的七个关节目标保持绝对关节角，
夹爪保持 `-1/+1` 命令。不要把 OpenPI 在训练 transform 后得到的 delta-action 统计量
用于这个 absolute-action LingBot checkpoint。

## LingBot-ready 目录

```text
robomme_data_lerobot/
├── empty_emb.pt
├── meta/
│   ├── info.json
│   ├── episodes.jsonl              # 每行额外含 action_config
│   ├── episodes_stats.jsonl
│   └── tasks.jsonl
├── data/
│   └── chunk-NNN/episode_NNNNNN.parquet
└── latents/
    └── chunk-NNN/
        ├── image/
        │   └── episode_NNNNNN_START_END.pth
        └── wrist_image/
            └── episode_NNNNNN_START_END.pth
```

每个 `action_config` segment 最少包含：

```json
{
  "start_frame": 0,
  "end_frame": 96,
  "action_text": "watch the video carefully, then ..."
}
```

`end_frame` 是左闭右开区间的终点，整段 episode 时等于 `length`。两路相机必须为
同一个 segment 各生成一个 latent 文件。每个 `.pth` 至少包含：

```text
latent, latent_num_frames, latent_height, latent_width,
video_num_frames, video_height, video_width,
text_emb, text, frame_ids, start_frame, end_frame, fps, ori_fps
```

`frame_ids` 必须来自原 episode 的 frame index，至少包含两个元素；loader 使用它的
首帧和 stride 从 Parquet 中截取、对齐 action。

## 准备与校验流程

以下命令从 `lingbot-va/` 执行。

0. 先把官方数据下载到独立的数据目录。路径不需要写死在仓库中，训练配置通过
   `ROBOMME_LEROBOT_DATASET_PATH` 读取。不要把数十 GB 级数据提交进源码目录。

```bash
huggingface-cli download Yinpei/robomme_data_lerobot \
  --repo-type dataset \
  --local-dir /data/robomme_data_lerobot

export ROBOMME_LEROBOT_DATASET_PATH=/data/robomme_data_lerobot
```

Windows PowerShell 可使用：

```powershell
$env:ROBOMME_LEROBOT_DATASET_PATH = 'E:\Science\Embodied_AI\data\robomme_data_lerobot'
```

1. 下载完成后，首先检查官方 raw dataset。`--scan-rows` 会逐帧检查两路 RGB 非空、action 为
   finite 8D、episode/frame index 正确、timestamp 与 10 FPS 一致。

```bash
python -m wan_va.utils.validate_robomme_lerobot \
  --dataset-root /path/to/robomme_data_lerobot \
  --scan-rows
```

2. 为每个 episode 生成最小的一段式 `action_config`。原文件会先产生带时间戳的备份。

```bash
python -m wan_va.utils.add_robomme_action_config \
  --dataset-root /path/to/robomme_data_lerobot \
  --in-place
```

如果 RoboMME 的 subgoal 要作为多段训练文本，应在生成 latent 之前用精确边界替换
一段式配置；不能只改 loader，因为 latent 文件名、`text_emb` 和 action 截取必须使用
同一组边界。

3. 使用本仓库的 RoboMME extractor 为 `image`、`wrist_image` 两个 Parquet 图像流
   提取 Wan2.2 latent，并生成 `empty_emb.pt`。模型目录必须是完整下载到本地的
   `robbyant/lingbot-va-base`，且包含 `vae/`、`tokenizer/` 和 `text_encoder/`。

```bash
export LINGBOT_WAN_MODEL_PATH=/path/to/lingbot-va-base
python wan_va/utils/extract_robomme_latents.py \
  --dataset-root /path/to/robomme_data_lerobot \
  --device cuda \
  --text-encoder-device cpu \
  --temporal-chunk-size 32
```

默认保持官方数据的 10 FPS，不覆盖已经存在的文件，因此任务中断后可直接重复执行。
正式运行前可以增加 `--dry-run` 只检查 metadata 和待生成文件。多卡/多机时可使用
`--episode-start` 与 `--episode-end`（右开）手动切分不重叠的 episode 范围。

extractor 会直接解码 Parquet 中两个 `dtype=image` 列，不需要先导出 MP4；同一 segment
内部使用 causal VAE 分块编码并保持时序 cache，在相机或 segment 边界清空 cache。

4. 从相同训练 split 的原始 `actions` 列计算 absolute-action quantiles。

```bash
python -m wan_va.utils.compute_robomme_action_stats \
  /path/to/robomme_data_lerobot \
  --output /path/to/robomme_absolute_action_stats.json
```

5. 做最终 LingBot-ready 校验。

```bash
export LINGBOT_ROBOMME_ACTION_STATS=/path/to/robomme_absolute_action_stats.json
python -m wan_va.utils.validate_robomme_lerobot \
  --dataset-root /path/to/robomme_data_lerobot \
  --scan-rows \
  --inspect-latents \
  --require-lingbot-ready
```

只有当输出同时为下面两行时，才应启动 `robomme_train`：

```text
Raw LeRobot v2.1 compatibility: PASS
LingBot training readiness:      PASS
```

6. 显式选择 RoboMME 后训练配置启动训练；启动脚本的默认配置仍是
   `robotwin_train`。

```bash
NGPU=8 CONFIG_NAME=robomme_train bash script/run_va_posttrain.sh
```
