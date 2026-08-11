# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
# 定义8D通道，action representation、q01/q99
import json
import os

from easydict import EasyDict

from .shared_config import va_shared_cfg

va_robomme_cfg = EasyDict(__name__='Config: VA RoboMME')
va_robomme_cfg.update(va_shared_cfg)
va_robomme_cfg.infer_mode = 'server'

va_robomme_cfg.wan22_pretrained_model_name_or_path = "/DATA/disk0/yry/lingbot-va-posttrain"

va_robomme_cfg.attn_window = 30
va_robomme_cfg.frame_chunk_size = 4
va_robomme_cfg.env_type = 'none'

va_robomme_cfg.height = 256
va_robomme_cfg.width = 256
va_robomme_cfg.action_dim = 30
va_robomme_cfg.action_per_frame = 4
# 官方 Yinpei/robomme_data_lerobot 数据集使用的字段名。
va_robomme_cfg.obs_cam_keys = ['image', 'wrist_image']
va_robomme_cfg.dataset_action_key = 'actions'
va_robomme_cfg.dataset_state_key = 'state'

# 官方 LeRobot action 已经是绝对关节目标：
# 格式为 [panda_j0, ..., panda_j6, gripper]。
va_robomme_cfg.action_representation = 'absolute_joint'
va_robomme_cfg.guidance_scale = 5
va_robomme_cfg.action_guidance_scale = 1

va_robomme_cfg.num_inference_steps = 20
va_robomme_cfg.video_exec_step = -1
va_robomme_cfg.action_num_inference_steps = 50

va_robomme_cfg.snr_shift = 5.0
va_robomme_cfg.action_snr_shift = 0.05

# RoboMME action 顺序：[panda_j0, ..., panda_j6, gripper]。
# LingBot canonical action 中的对应位置：
#   14:21 -> 左臂关节
#   28    -> 左夹爪
va_robomme_cfg.used_action_channel_ids = list(range(14, 21)) + [28]
inverse_used_action_channel_ids = [len(va_robomme_cfg.used_action_channel_ids)
                                   ] * va_robomme_cfg.action_dim
for i, j in enumerate(va_robomme_cfg.used_action_channel_ids):
    inverse_used_action_channel_ids[j] = i
va_robomme_cfg.inverse_used_action_channel_ids = inverse_used_action_channel_ids

va_robomme_cfg.action_norm_method = 'quantiles'


def _to_canonical_action_stat(values):
    """将 RoboMME 的 8D action 统计量映射到 LingBot 30D 布局。"""
    if len(values) != len(va_robomme_cfg.used_action_channel_ids):
        raise ValueError(
            f'Expected 8 RoboMME action statistics, got {len(values)}')
    canonical = [0.] * va_robomme_cfg.action_dim
    for channel_id, value in zip(va_robomme_cfg.used_action_channel_ids,
                                 values):
        canonical[channel_id] = value
    return canonical


def _load_action_quantiles(path):
    """加载 compute_robomme_action_stats.py 生成的 q01/q99。"""
    with open(path, 'r', encoding='utf-8') as file:
        payload = json.load(file)
    representation = payload.get('action_representation')
    if representation != va_robomme_cfg.action_representation:
        raise ValueError(
            f'Action stats {path!r} use representation {representation!r}; '
            f'expected {va_robomme_cfg.action_representation!r}')
    q01 = payload.get('q01')
    q99 = payload.get('q99')
    if q01 is None or q99 is None or len(q01) != 8 or len(q99) != 8:
        raise ValueError(f'Action stats {path!r} must contain 8D q01 and q99')
    return q01, q99


# 以下范围仅用于接口测试。正式训练/评测时，必须对实际使用的官方 LeRobot TODO
# split 运行 ``python -m wan_va.utils.compute_robomme_action_stats``，并通过
# LINGBOT_ROBOMME_ACTION_STATS 指定生成的 JSON。
robomme_action_q01 = [
    -0.34438448441028596,
    -0.22709672201871872,
    -0.28122754330635075,
    -2.860268020105362,
    -0.20585966855287552,
    1.5707963705062866,
    -0.4366058077812196,
    -1.0,
]
robomme_action_q99 = [
    0.3873456451892854,
    0.9852914290308954,
    0.27367080111503594,
    -1.161573127269745,
    0.20890909433364868,
    3.0531936925649643,
    2.1071231347084045,
    1.0,
]

va_robomme_cfg.action_stats_path = os.getenv(
    'LINGBOT_ROBOMME_ACTION_STATS', '')
if va_robomme_cfg.action_stats_path:
    robomme_action_q01, robomme_action_q99 = _load_action_quantiles(
        va_robomme_cfg.action_stats_path)

va_robomme_cfg.norm_stat = {
    'q01': _to_canonical_action_stat(robomme_action_q01),
    'q99': _to_canonical_action_stat(robomme_action_q99),
}
