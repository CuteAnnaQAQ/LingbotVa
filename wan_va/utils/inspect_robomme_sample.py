"""随机读取一个真实 RoboMME loader 样本并校验训练张量。"""

from __future__ import annotations

import argparse
import random

import torch

from wan_va.configs import VA_CONFIGS

try:
    from wan_va.dataset import MultiLatentLeRobotDataset
except ModuleNotFoundError as error:
    if error.name == "lerobot":
        raise RuntimeError(
            "当前 Python 环境缺少 lerobot；请运行 "
            "`python -m pip install lerobot==0.3.3 --no-deps`"
        ) from error
    raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = VA_CONFIGS["robomme_train"]
    torch.manual_seed(args.seed)
    dataset = MultiLatentLeRobotDataset(config, num_init_worker=1)
    index = (
        args.index
        if args.index is not None
        else random.Random(args.seed).randrange(len(dataset))
    )
    if not 0 <= index < len(dataset):
        raise ValueError(f"index {index} 超出数据集范围 [0, {len(dataset)})")

    dataset_id = dataset.item_id_to_dataset_id[index]
    inner_dataset = dataset._datasets[dataset_id]
    local_index = index - dataset.acc_dset_num[dataset_id]
    meta = inner_dataset.new_metas[local_index]
    latent_data = inner_dataset._get_range_latent_data(
        meta["start_frame"], meta["end_frame"], meta["episode_index"]
    )
    frame_ids = latent_data[f"{config.obs_cam_keys[0]}.frame_ids"]
    sample = dataset[index]

    latents = sample["latents"]
    text_emb = sample["text_emb"]
    actions = sample["actions"]
    action_mask = sample["actions_mask"]

    print(f"dataset length: {len(dataset)}")
    print(f"sample index: {index}")
    print(f"episode: {meta['episode_index']}")
    print(f"segment: [{meta['start_frame']}, {meta['end_frame']})")
    print(f"latents.shape: {tuple(latents.shape)}")
    print(f"text_emb.shape: {tuple(text_emb.shape)}")
    print(f"actions.shape: {tuple(actions.shape)}")
    print(f"action_mask.shape: {tuple(action_mask.shape)}")
    print(f"action min/max: {actions.min().item():.6f} / {actions.max().item():.6f}")
    print(f"frame_ids: {frame_ids}")

    if not all(torch.isfinite(value).all() for value in (latents, text_emb, actions)):
        raise ValueError("样本包含 NaN 或 Inf")
    if actions.min().item() < -1.5001 or actions.max().item() > 1.5001:
        raise ValueError("归一化 action 超出 [-1.5, 1.5]")
    if latents.ndim != 4 or text_emb.ndim != 2:
        raise ValueError(
            f"latent/text shape 不正确：{latents.shape} / {text_emb.shape}"
        )
    if actions.shape != action_mask.shape or actions.shape[0] != config.action_dim:
        raise ValueError(
            f"action/mask shape 不正确：{actions.shape} / {action_mask.shape}"
        )
    expected_latent_frames = (len(frame_ids) - 1) // 4 + 1
    if latents.shape[1] != expected_latent_frames:
        raise ValueError(
            f"latent 时间维 {latents.shape[1]} 与 frame_ids 推导值 "
            f"{expected_latent_frames} 不一致"
        )
    if actions.ndim != 4 or actions.shape[1:] != (
        expected_latent_frames,
        config.action_per_frame,
        1,
    ):
        raise ValueError(f"action 时序布局不正确：{actions.shape}")
    if action_mask.dtype != torch.bool:
        raise ValueError(f"action_mask 必须是 bool，实际为 {action_mask.dtype}")

    mask_by_channel = action_mask.reshape(config.action_dim, -1).any(dim=1)
    valid_channels = torch.where(mask_by_channel)[0].tolist()
    invalid_channels = torch.where(~mask_by_channel)[0].tolist()
    expected_valid = list(config.used_action_channel_ids)
    expected_invalid = sorted(set(range(config.action_dim)) - set(expected_valid))

    print(f"valid channels: {valid_channels}")
    print(f"invalid channels: {invalid_channels}")
    if valid_channels != expected_valid or invalid_channels != expected_invalid:
        raise ValueError(
            f"30D action mask 映射错误，期望有效通道 {expected_valid}，"
            f"实际为 {valid_channels}"
        )
    if not action_mask[expected_valid].all():
        raise ValueError("有效 action 通道中存在未启用的 mask 位置")
    if torch.count_nonzero(actions[invalid_channels]).item() != 0:
        raise ValueError("无效 action 通道中存在非零值")
    if len(frame_ids) < 2:
        raise ValueError("frame_ids 至少需要两个元素")
    frame_strides = {right - left for left, right in zip(frame_ids, frame_ids[1:])}
    if len(frame_strides) != 1 or next(iter(frame_strides)) <= 0:
        raise ValueError(f"frame_ids 不是严格等间隔递增：{frame_ids}")

    print("RoboMME loader single-sample check: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
