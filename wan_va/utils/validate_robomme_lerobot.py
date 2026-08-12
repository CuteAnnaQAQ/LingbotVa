"""校验 RoboMME LeRobot v2.1 数据和 LingBot 训练产物。

RoboMME 官方数据把 ``image`` 和 ``wrist_image`` 作为 Parquet 内嵌图像列
（``dtype: image``）保存，因此不要求 MP4。原始数据符合 LeRobot 规范，并不等于
已经满足 LingBot 训练条件；LingBot 还需要 ``action_config``、每路相机的 Wan2.2
latent、``empty_emb.pt`` 和 action 归一化统计量。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# 直接运行本文件时，避免同目录的 logging.py 遮蔽 Python 标准库 logging。
if __package__ in (None, ""):
    _script_directory = Path(__file__).resolve().parent
    sys.path[:] = [
        entry
        for entry in sys.path
        if Path(entry or os.getcwd()).resolve() != _script_directory
    ]


REQUIRED_FRAME_FEATURES = (
    "image",
    "wrist_image",
    "actions",
    "timestamp",
    "frame_index",
    "episode_index",
)
REQUIRED_LATENT_KEYS = (
    "latent",
    "latent_num_frames",
    "latent_height",
    "latent_width",
    "video_num_frames",
    "video_height",
    "video_width",
    "text_emb",
    "text",
    "frame_ids",
    "start_frame",
    "end_frame",
    "fps",
    "ori_fps",
)
EXPECTED_TEXT_EMBEDDING_SHAPE = (512, 4096)


@dataclass
class AuditResult:
    raw_errors: list[str] = field(default_factory=list)
    ready_errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    episodes_checked: int = 0
    frames_checked: int = 0

    @property
    def raw_compatible(self) -> bool:
        return not self.raw_errors

    @property
    def lingbot_ready(self) -> bool:
        return self.raw_compatible and not self.ready_errors


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}: {exc}"
                ) from exc
    return rows


def _episode_chunk(episode_index: int, chunks_size: int) -> int:
    return episode_index // chunks_size


def _format_episode_path(template: str, episode_index: int, chunks_size: int,
                         video_key: str | None = None) -> Path:
    values = {
        "episode_chunk": _episode_chunk(episode_index, chunks_size),
        "episode_index": episode_index,
    }
    if video_key is not None:
        values["video_key"] = video_key
    return Path(template.format(**values))


def _scalar(value: Any) -> Any:
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    return value


def _valid_rgb_payload(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, dict):
        return bool(value.get("bytes") or value.get("path"))
    # 部分 Arrow/datasets 版本会直接返回内嵌的二进制图像数据。
    if isinstance(value, (bytes, bytearray, memoryview, str)):
        return len(value) > 0
    return True


def _validate_info(info: dict[str, Any], result: AuditResult) -> None:
    if info.get("codebase_version") != "v2.1":
        result.raw_errors.append(
            "meta/info.json codebase_version must be 'v2.1' for lerobot==0.3.3"
        )

    for key in ("total_episodes", "total_frames", "fps", "chunks_size",
                "data_path", "features"):
        if key not in info:
            result.raw_errors.append(f"meta/info.json is missing {key!r}")

    features = info.get("features", {})
    for key in REQUIRED_FRAME_FEATURES:
        if key not in features:
            result.raw_errors.append(
                f"meta/info.json features is missing required key {key!r}")

    for key in ("image", "wrist_image"):
        feature = features.get(key, {})
        if feature.get("dtype") not in ("image", "video"):
            result.raw_errors.append(
                f"{key} must have dtype 'image' or 'video', got "
                f"{feature.get('dtype')!r}")
        shape = feature.get("shape")
        if not (isinstance(shape, list) and len(shape) == 3
                and (shape[0] == 3 or shape[-1] == 3)):
            result.raw_errors.append(
                f"{key} must be RGB with a 3-channel shape, got {shape!r}")

    action = features.get("actions", {})
    if action.get("dtype") not in ("float32", "float64"):
        result.raw_errors.append(
            f"actions must be floating point, got {action.get('dtype')!r}")
    if action.get("shape") != [8]:
        result.raw_errors.append(
            f"actions must be 8D [joint0..joint6, gripper], got "
            f"{action.get('shape')!r}")


def _validate_meta_rows(root: Path, info: dict[str, Any], result: AuditResult
                        ) -> tuple[list[dict[str, Any]], set[int]]:
    required = {
        "episodes": root / "meta" / "episodes.jsonl",
        "tasks": root / "meta" / "tasks.jsonl",
        "episodes_stats": root / "meta" / "episodes_stats.jsonl",
    }
    for name, path in required.items():
        if not path.is_file():
            result.raw_errors.append(
                f"LeRobot v2.1 requires meta/{name}.jsonl")
    if any(not path.is_file() for path in required.values()):
        return [], set()

    try:
        episodes = _read_jsonl(required["episodes"])
        tasks = _read_jsonl(required["tasks"])
        episodes_stats = _read_jsonl(required["episodes_stats"])
    except (OSError, ValueError) as exc:
        result.raw_errors.append(str(exc))
        return [], set()

    expected_episode_count = info.get("total_episodes")
    if expected_episode_count is not None and len(episodes) != expected_episode_count:
        result.raw_errors.append(
            f"episodes.jsonl has {len(episodes)} rows; info.json declares "
            f"{expected_episode_count}")

    episode_indices = [row.get("episode_index") for row in episodes]
    if episode_indices != list(range(len(episodes))):
        result.raw_errors.append(
            "episodes.jsonl episode_index values must be unique and contiguous "
            "from 0")

    for row in episodes:
        episode_index = row.get("episode_index")
        length = row.get("length")
        task_texts = row.get("tasks")
        if not isinstance(length, int) or length <= 0:
            result.raw_errors.append(
                f"episode {episode_index}: length must be a positive integer")
        if not (isinstance(task_texts, list) and task_texts
                and all(isinstance(text, str) and text.strip()
                        for text in task_texts)):
            result.raw_errors.append(
                f"episode {episode_index}: tasks must contain a language instruction")

    task_indices = {row.get("task_index") for row in tasks}
    if None in task_indices or len(task_indices) != len(tasks):
        result.raw_errors.append(
            "tasks.jsonl task_index values must be present and unique")
    if info.get("total_tasks") is not None and len(tasks) != info["total_tasks"]:
        result.raw_errors.append(
            f"tasks.jsonl has {len(tasks)} rows; info.json declares "
            f"{info['total_tasks']}")

    stats_indices = {row.get("episode_index") for row in episodes_stats}
    if stats_indices != set(range(len(episodes))):
        result.raw_errors.append(
            "episodes_stats.jsonl must contain one row for every episode_index")

    return episodes, {int(v) for v in task_indices if isinstance(v, int)}


def _validate_paths(root: Path, info: dict[str, Any],
                    episodes: list[dict[str, Any]], result: AuditResult) -> None:
    data_template = info.get("data_path")
    chunks_size = info.get("chunks_size")
    if not isinstance(data_template, str) or not isinstance(chunks_size, int):
        return

    features = info.get("features", {})
    video_keys = [
        key for key in ("image", "wrist_image")
        if features.get(key, {}).get("dtype") == "video"
    ]
    video_template = info.get("video_path")

    for episode in episodes:
        episode_index = episode["episode_index"]
        data_path = root / _format_episode_path(
            data_template, episode_index, chunks_size)
        if not data_path.is_file():
            result.raw_errors.append(
                f"episode {episode_index}: missing Parquet file {data_path}")
        if video_keys:
            if not isinstance(video_template, str):
                result.raw_errors.append(
                    "video features require info.json.video_path")
                return
            for key in video_keys:
                video_path = root / _format_episode_path(
                    video_template, episode_index, chunks_size, key)
                if not video_path.is_file():
                    result.raw_errors.append(
                        f"episode {episode_index}: missing {key} video {video_path}")

    if not video_keys:
        result.notes.append(
            "image and wrist_image use dtype=image; RGB is embedded in Parquet "
            "and videos/ is not required")


def _validate_parquet_rows(root: Path, info: dict[str, Any],
                           episodes: list[dict[str, Any]], task_indices: set[int],
                           result: AuditResult) -> None:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        result.raw_errors.append(
            "--scan-rows requires pyarrow; run this command inside the "
            "lerobot==0.3.3 environment")
        return

    data_template = info["data_path"]
    chunks_size = info["chunks_size"]
    fps = float(info["fps"])
    required_columns = list(REQUIRED_FRAME_FEATURES) + ["task_index"]

    for episode in episodes:
        episode_index = episode["episode_index"]
        path = root / _format_episode_path(
            data_template, episode_index, chunks_size)
        if not path.is_file():
            continue
        parquet_file = pq.ParquetFile(path)
        schema_names = set(parquet_file.schema_arrow.names)
        missing_columns = [key for key in required_columns if key not in schema_names]
        if missing_columns:
            result.raw_errors.append(
                f"episode {episode_index}: Parquet missing columns {missing_columns}")
            continue
        if parquet_file.metadata.num_rows != episode["length"]:
            result.raw_errors.append(
                f"episode {episode_index}: Parquet has "
                f"{parquet_file.metadata.num_rows} rows; episodes.jsonl length "
                f"is {episode['length']}")

        table = parquet_file.read(columns=required_columns)
        rows = {key: table.column(key).to_pylist() for key in required_columns}
        frame_count = table.num_rows
        result.episodes_checked += 1
        result.frames_checked += frame_count

        for frame_index in range(frame_count):
            prefix = f"episode {episode_index}, frame {frame_index}"
            if not _valid_rgb_payload(rows["image"][frame_index]):
                result.raw_errors.append(f"{prefix}: front RGB is empty")
            if not _valid_rgb_payload(rows["wrist_image"][frame_index]):
                result.raw_errors.append(f"{prefix}: wrist RGB is empty")

            action = rows["actions"][frame_index]
            if not (isinstance(action, (list, tuple)) and len(action) == 8
                    and all(isinstance(v, (int, float)) and math.isfinite(v)
                            for v in action)):
                result.raw_errors.append(f"{prefix}: actions is not finite 8D")

            stored_frame = _scalar(rows["frame_index"][frame_index])
            stored_episode = _scalar(rows["episode_index"][frame_index])
            timestamp = _scalar(rows["timestamp"][frame_index])
            task_index = _scalar(rows["task_index"][frame_index])
            if stored_frame != frame_index:
                result.raw_errors.append(
                    f"{prefix}: frame_index is {stored_frame!r}")
            if stored_episode != episode_index:
                result.raw_errors.append(
                    f"{prefix}: episode_index is {stored_episode!r}")
            if not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp):
                result.raw_errors.append(f"{prefix}: timestamp is not finite")
            elif abs(float(timestamp) - frame_index / fps) > 1e-4:
                result.raw_errors.append(
                    f"{prefix}: timestamp {timestamp} is not frame_index/fps")
            if task_index not in task_indices:
                result.raw_errors.append(
                    f"{prefix}: unknown task_index {task_index!r}")


def _validate_action_stats(path: Path | None, result: AuditResult) -> None:
    if path is None:
        result.ready_errors.append(
            "set --action-stats or LINGBOT_ROBOMME_ACTION_STATS to exact "
            "absolute_joint q01/q99 statistics")
        return
    if not path.is_file():
        result.ready_errors.append(f"action statistics file does not exist: {path}")
        return
    try:
        stats = _read_json(path)
    except (OSError, ValueError) as exc:
        result.ready_errors.append(str(exc))
        return
    if stats.get("action_representation") != "absolute_joint":
        result.ready_errors.append(
            "action statistics must declare action_representation=absolute_joint")
    for key in ("q01", "q99"):
        values = stats.get(key)
        if not (isinstance(values, list) and len(values) == 8
                and all(isinstance(v, (int, float)) and math.isfinite(v)
                        for v in values)):
            result.ready_errors.append(f"action statistics {key} must be finite 8D")


def _validate_text_embedding(value: Any, label: str,
                             result: AuditResult) -> None:
    shape = getattr(value, "shape", None)
    actual_shape = tuple(shape) if shape is not None else None
    if actual_shape != EXPECTED_TEXT_EMBEDDING_SHAPE:
        result.ready_errors.append(
            f"{label}: text embedding shape must be "
            f"{EXPECTED_TEXT_EMBEDDING_SHAPE}, got {actual_shape}")


def _validate_lingbot_ready(root: Path, info: dict[str, Any],
                            episodes: list[dict[str, Any]], result: AuditResult,
                            inspect_latents: bool, action_stats: Path | None) -> None:
    empty_embedding_path = root / "empty_emb.pt"
    if not empty_embedding_path.is_file():
        result.ready_errors.append(
            f"missing LingBot empty embedding: {empty_embedding_path}")
    _validate_action_stats(action_stats, result)

    chunks_size = info.get("chunks_size")
    if not isinstance(chunks_size, int):
        return
    latent_records: list[tuple[Path, int, int, int, str]] = []
    for episode in episodes:
        episode_index = episode.get("episode_index")
        length = episode.get("length")
        configs = episode.get("action_config")
        if not isinstance(configs, list) or not configs:
            result.ready_errors.append(
                f"episode {episode_index}: missing non-empty action_config")
            continue
        for segment in configs:
            start = segment.get("start_frame")
            end = segment.get("end_frame")
            text = segment.get("action_text")
            if not (isinstance(start, int) and isinstance(end, int)
                    and isinstance(length, int) and 0 <= start < end <= length):
                result.ready_errors.append(
                    f"episode {episode_index}: invalid action_config bounds "
                    f"[{start}, {end}) for length {length}")
                continue
            if not isinstance(text, str) or not text.strip():
                result.ready_errors.append(
                    f"episode {episode_index}: action_text must be non-empty")
            chunk = _episode_chunk(episode_index, chunks_size)
            for camera_key in ("image", "wrist_image"):
                path = (root / "latents" / f"chunk-{chunk:03d}" / camera_key /
                        f"episode_{episode_index:06d}_{start}_{end}.pth")
                if not path.is_file():
                    result.ready_errors.append(
                        f"episode {episode_index}: missing {camera_key} latent {path}")
                else:
                    latent_records.append((path, episode_index, start, end, camera_key))

    if not inspect_latents:
        return
    try:
        import torch
    except ImportError:
        result.ready_errors.append(
            "--inspect-latents requires torch inside the LingBot environment")
        return
    if empty_embedding_path.is_file():
        empty_embedding = torch.load(
            empty_embedding_path, map_location="cpu", weights_only=False)
        _validate_text_embedding(
            empty_embedding, str(empty_embedding_path), result)
    for path, episode_index, start, end, camera_key in latent_records:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        missing = [key for key in REQUIRED_LATENT_KEYS if key not in payload]
        if missing:
            result.ready_errors.append(
                f"{path}: missing latent fields {missing}")
            continue
        if payload["start_frame"] != start or payload["end_frame"] != end:
            result.ready_errors.append(
                f"{path}: latent segment bounds do not match action_config")
        if not payload["frame_ids"] or len(payload["frame_ids"]) < 2:
            result.ready_errors.append(
                f"{path}: {camera_key} frame_ids must contain at least 2 frames")
        _validate_text_embedding(payload["text_emb"], str(path), result)


def audit_dataset(dataset_root: str | Path, *, scan_rows: bool = False,
                  inspect_latents: bool = False,
                  action_stats: str | Path | None = None) -> AuditResult:
    root = Path(dataset_root)
    result = AuditResult()
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        result.raw_errors.append(f"missing metadata: {info_path}")
        return result
    try:
        info = _read_json(info_path)
    except (OSError, ValueError) as exc:
        result.raw_errors.append(str(exc))
        return result

    _validate_info(info, result)
    episodes, task_indices = _validate_meta_rows(root, info, result)
    _validate_paths(root, info, episodes, result)
    if scan_rows and episodes:
        _validate_parquet_rows(root, info, episodes, task_indices, result)

    stats_value = action_stats or os.getenv("LINGBOT_ROBOMME_ACTION_STATS")
    stats_path = Path(stats_value) if stats_value else None
    _validate_lingbot_ready(
        root, info, episodes, result, inspect_latents, stats_path)
    return result


def _print_result(result: AuditResult, require_lingbot_ready: bool) -> int:
    raw_label = "PASS" if result.raw_compatible else "FAIL"
    ready_label = "PASS" if result.lingbot_ready else "FAIL"
    print(f"Raw LeRobot v2.1 compatibility: {raw_label}")
    print(f"LingBot training readiness:      {ready_label}")
    if result.episodes_checked:
        print(
            f"Full row scan: {result.episodes_checked} episodes, "
            f"{result.frames_checked} frames")
    for note in result.notes:
        print(f"[NOTE] {note}")
    max_messages = 30
    for error in result.raw_errors[:max_messages]:
        print(f"[RAW ERROR] {error}")
    if len(result.raw_errors) > max_messages:
        print(f"[RAW ERROR] ... {len(result.raw_errors) - max_messages} more")
    for error in result.ready_errors[:max_messages]:
        level = "READY ERROR" if require_lingbot_ready else "READY TODO"
        print(f"[{level}] {error}")
    if len(result.ready_errors) > max_messages:
        level = "READY ERROR" if require_lingbot_ready else "READY TODO"
        print(f"[{level}] ... {len(result.ready_errors) - max_messages} more")
    return int(not result.raw_compatible
               or (require_lingbot_ready and not result.lingbot_ready))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="检查 RoboMME LeRobot v2.1 数据及 LingBot latent 完整性")
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument(
        "--scan-rows", action="store_true",
        help="逐行检查 Parquet 中的 RGB、action、索引和时间戳")
    parser.add_argument(
        "--require-lingbot-ready", action="store_true",
        help="action_config、latent 或统计量不完整时返回非零状态码")
    parser.add_argument(
        "--inspect-latents", action="store_true",
        help="通过 torch.load 检查每个 latent 文件的字段")
    parser.add_argument(
        "--action-stats", type=Path,
        help="8D 统计量 JSON；默认读取 LINGBOT_ROBOMME_ACTION_STATS")
    args = parser.parse_args()
    result = audit_dataset(
        args.dataset_root,
        scan_rows=args.scan_rows,
        inspect_latents=args.inspect_latents,
        action_stats=args.action_stats,
    )
    return _print_result(result, args.require_lingbot_ready)


if __name__ == "__main__":
    raise SystemExit(main())
