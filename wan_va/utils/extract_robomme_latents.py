"""把 RoboMME LeRobot v2.1 的 Parquet 图像离线编码为 LingBot latent。

官方 RoboMME 数据把 ``image`` 和 ``wrist_image`` 作为内嵌图像存放在
Parquet 中。本工具按 ``meta/episodes.jsonl`` 的 ``action_config`` 分段，使用
LingBot 基座模型中的 Wan2.2 causal VAE 和 UMT5 编码器生成旧版
``MultiLatentLeRobotDataset`` 所需的逐相机 ``.pth`` 文件。

默认不会覆盖已有文件，因此中断后可直接重复执行。只有显式传入 ``--force``
才会覆盖已有 latent 和 ``empty_emb.pt``。
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


# 直接运行本文件时，移除 utils 目录，避免其中的 logging.py 遮蔽标准库 logging。
if __package__ in (None, ""):
    _script_directory = Path(__file__).resolve().parent
    sys.path[:] = [
        entry
        for entry in sys.path
        if Path(entry or os.getcwd()).resolve() != _script_directory
    ]


TEMPORAL_DOWNSAMPLE = 4
DEFAULT_CAMERA_KEYS = ("image", "wrist_image")


def _info(message: str, *args: Any) -> None:
    print(message % args if args else message, file=sys.stderr)


def _error(message: str, *args: Any) -> None:
    print(f"ERROR: {message % args if args else message}", file=sys.stderr)


@dataclass(frozen=True)
class SegmentJob:
    episode_index: int
    episode_chunk: int
    episode_length: int
    start_frame: int
    end_frame: int
    action_text: str


@dataclass
class ModelBundle:
    torch: Any
    functional: Any
    vae: Any
    streaming_vae: Any
    tokenizer: Any
    text_encoder: Any
    dtype: Any
    vae_device: Any
    text_device: Any
    prompt_clean: Any


class _WanVAEStreamingEncoder:
    """最小化复用 LingBot 的 causal VAE 分块编码协议。"""

    def __init__(self, vae: Any):
        self.vae = vae
        self.encoder = vae.encoder
        self.quant_conv = vae.quant_conv
        if hasattr(vae, "_cached_conv_counts"):
            self.encoder_cache_count = vae._cached_conv_counts["encoder"]
        else:
            self.encoder_cache_count = sum(
                module.__class__.__name__ == "WanCausalConv3d"
                for module in self.encoder.modules()
            )
        self.clear_cache()

    def clear_cache(self) -> None:
        self.feature_cache = [None] * self.encoder_cache_count

    @staticmethod
    def _patchify(value: Any, patch_size: int | None) -> Any:
        if patch_size is None or patch_size == 1:
            return value
        batch, channels, frames, height, width = value.shape
        if height % patch_size or width % patch_size:
            raise ValueError(
                f"VAE patch_size={patch_size} does not divide "
                f"input size {(height, width)}"
            )
        value = value.view(
            batch,
            channels,
            frames,
            height // patch_size,
            patch_size,
            width // patch_size,
            patch_size,
        )
        value = value.permute(0, 1, 6, 4, 2, 3, 5).contiguous()
        return value.view(
            batch,
            channels * patch_size * patch_size,
            frames,
            height // patch_size,
            width // patch_size,
        )

    def encode_chunk(self, value: Any) -> Any:
        value = self._patchify(
            value, getattr(self.vae.config, "patch_size", None)
        )
        feature_index = [0]
        encoded = self.encoder(
            value,
            feat_cache=self.feature_cache,
            feat_idx=feature_index,
        )
        return self.quant_conv(encoded)


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


def _scalar(value: Any) -> Any:
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    return value


def _episode_chunk(episode_index: int, chunks_size: int) -> int:
    return episode_index // chunks_size


def _format_episode_path(
    template: str, episode_index: int, chunks_size: int
) -> Path:
    return Path(
        template.format(
            episode_chunk=_episode_chunk(episode_index, chunks_size),
            episode_index=episode_index,
        )
    )


def _latent_path(root: Path, camera_key: str, job: SegmentJob) -> Path:
    return (
        root
        / "latents"
        / f"chunk-{job.episode_chunk:03d}"
        / camera_key
        / (
            f"episode_{job.episode_index:06d}_"
            f"{job.start_frame}_{job.end_frame}.pth"
        )
    )


def _validate_info(info: dict[str, Any], camera_keys: Sequence[str]) -> None:
    if info.get("codebase_version") != "v2.1":
        raise ValueError(
            "RoboMME extractor requires LeRobot codebase_version='v2.1'"
        )
    for key in ("fps", "chunks_size", "data_path", "features"):
        if key not in info:
            raise ValueError(f"meta/info.json is missing {key!r}")
    if not isinstance(info["chunks_size"], int) or info["chunks_size"] <= 0:
        raise ValueError("meta/info.json chunks_size must be a positive integer")
    if not isinstance(info["fps"], (int, float)) or info["fps"] <= 0:
        raise ValueError("meta/info.json fps must be positive")
    if not isinstance(info["data_path"], str):
        raise ValueError("meta/info.json data_path must be a string")
    features = info["features"]
    if not isinstance(features, dict):
        raise ValueError("meta/info.json features must be an object")
    for key in camera_keys:
        feature = features.get(key)
        if not isinstance(feature, dict):
            raise ValueError(f"meta/info.json features is missing {key!r}")
        if feature.get("dtype") != "image":
            raise ValueError(
                f"{key!r} must be an embedded Parquet image column; "
                f"got dtype={feature.get('dtype')!r}"
            )


def build_segment_jobs(
    episodes: Sequence[dict[str, Any]],
    chunks_size: int,
    *,
    episode_start: int = 0,
    episode_end: int | None = None,
) -> list[SegmentJob]:
    """校验 action_config，并构造指定 episode 范围内的提取任务。"""
    if episode_start < 0:
        raise ValueError("episode_start must be non-negative")
    if episode_end is not None and episode_end <= episode_start:
        raise ValueError("episode_end must be greater than episode_start")

    jobs = []
    for episode in episodes:
        episode_index = episode.get("episode_index")
        length = episode.get("length")
        if not isinstance(episode_index, int) or episode_index < 0:
            raise ValueError(f"invalid episode_index: {episode_index!r}")
        if episode_index < episode_start:
            continue
        if episode_end is not None and episode_index >= episode_end:
            continue
        if not isinstance(length, int) or length <= 0:
            raise ValueError(
                f"episode {episode_index}: length must be a positive integer"
            )
        segments = episode.get("action_config")
        if not isinstance(segments, list) or not segments:
            raise ValueError(
                f"episode {episode_index}: missing non-empty action_config; "
                "run wan_va.utils.add_robomme_action_config first"
            )
        for segment_index, segment in enumerate(segments):
            start = segment.get("start_frame")
            end = segment.get("end_frame")
            text = segment.get("action_text")
            if not (
                isinstance(start, int)
                and isinstance(end, int)
                and 0 <= start < end <= length
            ):
                raise ValueError(
                    f"episode {episode_index} segment {segment_index}: invalid "
                    f"bounds [{start}, {end}) for length {length}"
                )
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"episode {episode_index} segment {segment_index}: "
                    "action_text must be non-empty"
                )
            jobs.append(
                SegmentJob(
                    episode_index=episode_index,
                    episode_chunk=_episode_chunk(episode_index, chunks_size),
                    episode_length=length,
                    start_frame=start,
                    end_frame=end,
                    action_text=text.strip(),
                )
            )
    if not jobs:
        raise ValueError("no action_config segments selected")
    return jobs


def select_frame_ids(
    start_frame: int,
    end_frame: int,
    source_fps: float,
    target_fps: float | None,
) -> tuple[list[int], int, float]:
    """返回等间隔源帧索引；现有 loader 只支持整数 frame stride。"""
    if not (0 <= start_frame < end_frame):
        raise ValueError(f"invalid frame range [{start_frame}, {end_frame})")
    if not math.isfinite(source_fps) or source_fps <= 0:
        raise ValueError("source_fps must be positive and finite")
    requested_fps = source_fps if target_fps is None else target_fps
    if not math.isfinite(requested_fps) or requested_fps <= 0:
        raise ValueError("target_fps must be positive and finite")
    ratio = source_fps / requested_fps
    stride = int(round(ratio))
    if stride < 1 or not math.isclose(ratio, stride, rel_tol=0, abs_tol=1e-6):
        raise ValueError(
            f"source_fps/target_fps must be a positive integer because the "
            f"LingBot loader assumes a constant integer frame stride; got "
            f"{source_fps}/{requested_fps}={ratio}"
        )
    frame_ids = list(range(start_frame, end_frame, stride))
    if len(frame_ids) < 2:
        raise ValueError(
            f"segment [{start_frame}, {end_frame}) yields fewer than two "
            f"frames at stride {stride}"
        )
    return frame_ids, stride, source_fps / stride


def _component_paths(model_root: Path) -> dict[str, Path]:
    root = model_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"local LingBot model root does not exist: {root}; download "
            "robbyant/lingbot-va-base first"
        )
    paths = {
        "vae": root / "vae",
        "tokenizer": root / "tokenizer",
        "text_encoder": root / "text_encoder",
    }
    missing = [str(path) for path in paths.values() if not path.is_dir()]
    if missing:
        raise FileNotFoundError(
            f"LingBot model root is missing component directories: {missing}"
        )
    return paths


def _torch_dtype(torch: Any, name: str) -> Any:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping[name]


def _load_models(
    model_root: Path,
    *,
    device: str,
    text_encoder_device: str,
    dtype_name: str,
) -> ModelBundle:
    try:
        import torch
        import torch.nn.functional as functional
        from diffusers import AutoencoderKLWan
        from diffusers.pipelines.wan.pipeline_wan import prompt_clean
        from transformers import T5TokenizerFast, UMT5EncoderModel
    except ImportError as exc:
        raise RuntimeError(
            "latent extraction requires the full LingBot environment with "
            "torch, diffusers and transformers installed"
        ) from exc

    components = _component_paths(model_root)
    dtype = _torch_dtype(torch, dtype_name)
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")
    vae_device = torch.device(device)
    text_device = torch.device(text_encoder_device)
    if text_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA text encoder device requested but unavailable: "
            f"{text_encoder_device}"
        )

    _info("加载 Wan2.2 VAE：%s", components["vae"])
    vae = AutoencoderKLWan.from_pretrained(
        str(components["vae"]), torch_dtype=dtype
    ).to(vae_device).eval()
    vae.requires_grad_(False)

    _info("加载 UMT5 text encoder：%s", components["text_encoder"])
    text_encoder = UMT5EncoderModel.from_pretrained(
        str(components["text_encoder"]),
        torch_dtype=dtype,
    ).to(text_device).eval()
    text_encoder.requires_grad_(False)
    tokenizer = T5TokenizerFast.from_pretrained(str(components["tokenizer"]))

    return ModelBundle(
        torch=torch,
        functional=functional,
        vae=vae,
        streaming_vae=_WanVAEStreamingEncoder(vae),
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        dtype=dtype,
        vae_device=vae_device,
        text_device=text_device,
        prompt_clean=prompt_clean,
    )


def _encode_text(
    bundle: ModelBundle, text: str, max_sequence_length: int
) -> Any:
    torch = bundle.torch
    cleaned = bundle.prompt_clean(text)
    inputs = bundle.tokenizer(
        [cleaned],
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_ids = inputs.input_ids.to(bundle.text_device)
    attention_mask = inputs.attention_mask.to(bundle.text_device)
    with torch.inference_mode():
        hidden = bundle.text_encoder(
            input_ids, attention_mask=attention_mask
        ).last_hidden_state
    sequence_length = int(attention_mask[0].sum().item())
    hidden = hidden[0, :sequence_length]
    if sequence_length < max_sequence_length:
        hidden = bundle.functional.pad(
            hidden, (0, 0, 0, max_sequence_length - sequence_length)
        )
    return hidden.to(dtype=torch.bfloat16, device="cpu").contiguous()


def _decode_image_cell(
    value: Any, *, dataset_root: Path, parquet_path: Path
) -> Any:
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "image decoding requires numpy and Pillow inside the LingBot environment"
        ) from exc

    if isinstance(value, dict):
        embedded = value.get("bytes")
        if embedded is not None:
            value = embedded
        else:
            value = value.get("path")
    if isinstance(value, memoryview):
        value = value.tobytes()
    if isinstance(value, (bytes, bytearray)):
        image = Image.open(io.BytesIO(value))
    elif isinstance(value, (str, os.PathLike)):
        source = Path(value)
        candidates = [source]
        if not source.is_absolute():
            candidates = [dataset_root / source, parquet_path.parent / source]
        image_path = next((path for path in candidates if path.is_file()), None)
        if image_path is None:
            raise FileNotFoundError(
                f"embedded image path does not exist: {value!r}"
            )
        image = Image.open(image_path)
    elif isinstance(value, np.ndarray):
        array = value
        if array.ndim != 3:
            raise ValueError(f"expected an RGB image array, got {array.shape}")
        if array.shape[0] == 3 and array.shape[-1] != 3:
            array = array.transpose(1, 2, 0)
        return np.asarray(array, dtype=np.uint8).copy()
    else:
        raise TypeError(f"unsupported Parquet image payload: {type(value)!r}")
    with image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def _read_episode_columns(
    parquet_path: Path, camera_keys: Sequence[str]
) -> tuple[dict[str, list[Any]], dict[int, int]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "RoboMME Parquet extraction requires pyarrow inside the "
            "lerobot==0.3.3 environment"
        ) from exc

    parquet = pq.ParquetFile(parquet_path)
    required = ["frame_index", *camera_keys]
    schema_names = set(parquet.schema_arrow.names)
    missing = [key for key in required if key not in schema_names]
    if missing:
        raise ValueError(f"{parquet_path}: missing Parquet columns {missing}")
    table = parquet.read(columns=required)
    columns = {key: table.column(key).to_pylist() for key in camera_keys}
    frame_values = [_scalar(value) for value in table.column("frame_index").to_pylist()]
    frame_to_row: dict[int, int] = {}
    for row_index, frame_index in enumerate(frame_values):
        if not isinstance(frame_index, int):
            raise ValueError(
                f"{parquet_path}: frame_index row {row_index} is "
                f"{frame_index!r}, expected int"
            )
        if frame_index in frame_to_row:
            raise ValueError(
                f"{parquet_path}: duplicate frame_index {frame_index}"
            )
        frame_to_row[frame_index] = row_index
    return columns, frame_to_row


def _preprocess_frames(bundle: ModelBundle, frames: Sequence[Any], size: tuple[int, int]) -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("frame preprocessing requires numpy") from exc
    torch = bundle.torch
    array = np.stack(frames, axis=0)
    tensor = torch.from_numpy(array).permute(0, 3, 1, 2).float()
    tensor = bundle.functional.interpolate(
        tensor, size=size, mode="bilinear", align_corners=False
    )
    tensor = tensor.permute(1, 0, 2, 3).unsqueeze(0)
    tensor = tensor / 255.0 * 2.0 - 1.0
    return tensor.to(device=bundle.vae_device, dtype=bundle.dtype)


def _normalize_encoder_output(bundle: ModelBundle, encoder_output: Any) -> Any:
    torch = bundle.torch
    mu, _logvar = torch.chunk(encoder_output, 2, dim=1)
    mean = torch.tensor(
        bundle.vae.config.latents_mean, dtype=torch.float32, device=mu.device
    ).view(1, -1, 1, 1, 1)
    inverse_std = (
        1.0
        / torch.tensor(
            bundle.vae.config.latents_std,
            dtype=torch.float32,
            device=mu.device,
        )
    ).view(1, -1, 1, 1, 1)
    return ((mu.float() - mean) * inverse_std).to(mu)


def _encode_camera_segment(
    bundle: ModelBundle,
    image_values: Sequence[Any],
    frame_to_row: dict[int, int],
    frame_ids: Sequence[int],
    *,
    dataset_root: Path,
    parquet_path: Path,
    height: int,
    width: int,
    temporal_chunk_size: int,
) -> tuple[Any, int, int, int]:
    torch = bundle.torch
    missing = [frame_id for frame_id in frame_ids if frame_id not in frame_to_row]
    if missing:
        raise ValueError(
            f"{parquet_path}: selected frame indices are absent; first values: "
            f"{missing[:5]}"
        )

    bundle.streaming_vae.clear_cache()
    latent_chunks = []
    with torch.inference_mode():
        for offset in range(0, len(frame_ids), temporal_chunk_size):
            chunk_ids = frame_ids[offset : offset + temporal_chunk_size]
            frames = [
                _decode_image_cell(
                    image_values[frame_to_row[frame_id]],
                    dataset_root=dataset_root,
                    parquet_path=parquet_path,
                )
                for frame_id in chunk_ids
            ]
            video = _preprocess_frames(bundle, frames, (height, width))
            encoded = bundle.streaming_vae.encode_chunk(video)
            latent_chunks.append(
                _normalize_encoder_output(bundle, encoded).to("cpu")
            )

    latent = torch.cat(latent_chunks, dim=2)
    if latent.shape[0] != 1:
        raise ValueError(f"expected one encoded camera, got {latent.shape}")
    expected_frames = (len(frame_ids) - 1) // TEMPORAL_DOWNSAMPLE + 1
    if latent.shape[2] != expected_frames:
        raise ValueError(
            f"Wan2.2 VAE produced {latent.shape[2]} latent frames for "
            f"{len(frame_ids)} sampled frames; expected {expected_frames}. "
            "Use a temporal chunk size divisible by 4 and the matching "
            "LingBot base-model VAE."
        )
    _, channels, latent_frames, latent_height, latent_width = latent.shape
    flattened = (
        latent[0]
        .permute(1, 2, 3, 0)
        .reshape(latent_frames * latent_height * latent_width, channels)
        .to(dtype=torch.bfloat16)
        .contiguous()
    )
    return flattened, latent_frames, latent_height, latent_width


def _atomic_torch_save(torch: Any, payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _save_empty_embedding(
    bundle: ModelBundle,
    path: Path,
    *,
    max_sequence_length: int,
    force: bool,
) -> None:
    if path.is_file() and not force:
        _info("跳过已有空文本 embedding：%s", path)
        return
    embedding = _encode_text(bundle, "", max_sequence_length)
    _atomic_torch_save(bundle.torch, embedding, path)
    _info("写入空文本 embedding：%s", path)


def _segment_payload(
    *,
    latent: Any,
    latent_frames: int,
    latent_height: int,
    latent_width: int,
    text_embedding: Any,
    job: SegmentJob,
    frame_ids: Sequence[int],
    target_fps: float,
    source_fps: float,
    height: int,
    width: int,
) -> dict[str, Any]:
    stored_target_fps: int | float = (
        int(round(target_fps))
        if math.isclose(target_fps, round(target_fps), abs_tol=1e-6)
        else target_fps
    )
    stored_source_fps: int | float = (
        int(round(source_fps))
        if math.isclose(source_fps, round(source_fps), abs_tol=1e-6)
        else source_fps
    )
    return {
        "latent": latent,
        "latent_num_frames": latent_frames,
        "latent_height": latent_height,
        "latent_width": latent_width,
        "video_num_frames": len(frame_ids),
        "video_height": height,
        "video_width": width,
        "text_emb": text_embedding,
        "text": job.action_text,
        "frame_ids": list(frame_ids),
        "start_frame": job.start_frame,
        "end_frame": job.end_frame,
        "fps": stored_target_fps,
        "ori_fps": stored_source_fps,
    }


def _pending_camera_keys(
    root: Path, job: SegmentJob, camera_keys: Sequence[str], force: bool
) -> list[str]:
    if force:
        return list(camera_keys)
    return [key for key in camera_keys if not _latent_path(root, key, job).is_file()]


def extract_latents(args: argparse.Namespace) -> int:
    root = args.dataset_root.expanduser().resolve()
    info_path = root / "meta" / "info.json"
    episodes_path = root / "meta" / "episodes.jsonl"
    if not info_path.is_file():
        raise FileNotFoundError(f"metadata not found: {info_path}")
    if not episodes_path.is_file():
        raise FileNotFoundError(f"metadata not found: {episodes_path}")

    camera_keys = tuple(args.camera_keys)
    if len(set(camera_keys)) != len(camera_keys):
        raise ValueError(f"camera keys must be unique: {camera_keys}")
    info = _read_json(info_path)
    _validate_info(info, camera_keys)
    jobs = build_segment_jobs(
        _read_jsonl(episodes_path),
        info["chunks_size"],
        episode_start=args.episode_start,
        episode_end=args.episode_end,
    )
    source_fps = float(info["fps"])
    for job in jobs:
        select_frame_ids(
            job.start_frame, job.end_frame, source_fps, args.target_fps
        )

    pending_outputs = sum(
        len(_pending_camera_keys(root, job, camera_keys, args.force))
        for job in jobs
    )
    empty_path = (
        args.empty_embedding_path.expanduser().resolve()
        if args.empty_embedding_path is not None
        else root / "empty_emb.pt"
    )
    empty_pending = args.force or not empty_path.is_file()
    print(
        f"Selected {len(jobs)} segment(s), {pending_outputs} latent file(s) "
        f"pending, empty_emb pending={empty_pending}"
    )
    if args.dry_run:
        for job in jobs[:5]:
            pending = _pending_camera_keys(root, job, camera_keys, args.force)
            print(
                f"episode={job.episode_index} segment="
                f"[{job.start_frame},{job.end_frame}) pending={pending}"
            )
        return 0
    if pending_outputs == 0 and not empty_pending:
        print("All requested LingBot latent artifacts already exist.")
        return 0
    if args.model_path is None:
        raise ValueError(
            "set --model-path or LINGBOT_WAN_MODEL_PATH to a local "
            "robbyant/lingbot-va-base directory"
        )

    bundle = _load_models(
        args.model_path,
        device=args.device,
        text_encoder_device=args.text_encoder_device,
        dtype_name=args.dtype,
    )
    _save_empty_embedding(
        bundle,
        empty_path,
        max_sequence_length=args.max_sequence_length,
        force=args.force,
    )

    data_template = info["data_path"]
    chunks_size = info["chunks_size"]
    jobs_by_episode: dict[int, list[SegmentJob]] = {}
    for job in jobs:
        jobs_by_episode.setdefault(job.episode_index, []).append(job)

    written = 0
    skipped = 0
    for episode_number, (episode_index, episode_jobs) in enumerate(
        jobs_by_episode.items(), 1
    ):
        episode_pending = {
            job: _pending_camera_keys(root, job, camera_keys, args.force)
            for job in episode_jobs
        }
        if not any(episode_pending.values()):
            skipped += len(episode_jobs) * len(camera_keys)
            continue
        parquet_path = root / _format_episode_path(
            data_template, episode_index, chunks_size
        )
        if not parquet_path.is_file():
            raise FileNotFoundError(
                f"episode {episode_index}: Parquet file not found: {parquet_path}"
            )
        _info(
            "[%d/%d] 读取 episode %d：%s",
            episode_number,
            len(jobs_by_episode),
            episode_index,
            parquet_path,
        )
        columns, frame_to_row = _read_episode_columns(parquet_path, camera_keys)

        for job in episode_jobs:
            pending_keys = episode_pending[job]
            skipped += len(camera_keys) - len(pending_keys)
            if not pending_keys:
                continue
            frame_ids, _stride, actual_fps = select_frame_ids(
                job.start_frame, job.end_frame, source_fps, args.target_fps
            )
            text_embedding = _encode_text(
                bundle, job.action_text, args.max_sequence_length
            )
            for camera_key in pending_keys:
                _info(
                    "编码 episode %d [%d,%d) camera=%s frames=%d",
                    job.episode_index,
                    job.start_frame,
                    job.end_frame,
                    camera_key,
                    len(frame_ids),
                )
                latent, latent_frames, latent_height, latent_width = (
                    _encode_camera_segment(
                        bundle,
                        columns[camera_key],
                        frame_to_row,
                        frame_ids,
                        dataset_root=root,
                        parquet_path=parquet_path,
                        height=args.height,
                        width=args.width,
                        temporal_chunk_size=args.temporal_chunk_size,
                    )
                )
                payload = _segment_payload(
                    latent=latent,
                    latent_frames=latent_frames,
                    latent_height=latent_height,
                    latent_width=latent_width,
                    text_embedding=text_embedding,
                    job=job,
                    frame_ids=frame_ids,
                    target_fps=actual_fps,
                    source_fps=source_fps,
                    height=args.height,
                    width=args.width,
                )
                output_path = _latent_path(root, camera_key, job)
                _atomic_torch_save(bundle.torch, payload, output_path)
                written += 1
                _info("写入：%s", output_path)
        if bundle.vae_device.type == "cuda":
            bundle.torch.cuda.empty_cache()

    print(f"Extraction complete: wrote={written}, skipped={skipped}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="/DATA/disk0/yry/robomme_data_lerobot", type=Path)
    parser.add_argument(
        "--model-path",
        type=Path,
        default="/DATA/disk0/yry/lingbot-va-base",
    )
    parser.add_argument(
        "--camera-keys",
        nargs="+",
        default=list(DEFAULT_CAMERA_KEYS),
        help="要提取的 Parquet 图像列，默认：image wrist_image",
    )
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument(
        "--target-fps",
        type=float,
        default=None,
        help="目标 FPS；默认保持数据集原 FPS，且 source_fps/target_fps 必须为整数",
    )
    parser.add_argument(
        "--temporal-chunk-size",
        type=int,
        default=32,
        help="每次送入 causal VAE 的连续视频帧数，必须为 4 的倍数",
    )
    parser.add_argument("--max-sequence-length", type=int, default=226)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--text-encoder-device", default="cpu")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument(
        "--episode-end",
        type=int,
        default=None,
        help="exclusive；可用于多进程/多机手动切分 episode 范围",
    )
    parser.add_argument(
        "--empty-embedding-path",
        type=Path,
        default=None,
        help="默认写到 <dataset-root>/empty_emb.pt",
    )
    parser.add_argument(
        "--force", action="store_true", help="覆盖已有 latent 和 empty_emb.pt"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只校验元数据并打印待生成任务，不加载模型/Parquet",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.height <= 0 or args.width <= 0:
        parser.error("--height and --width must be positive")
    if (
        args.temporal_chunk_size < TEMPORAL_DOWNSAMPLE
        or args.temporal_chunk_size % TEMPORAL_DOWNSAMPLE != 0
    ):
        parser.error("--temporal-chunk-size must be a positive multiple of 4")
    if args.max_sequence_length <= 0:
        parser.error("--max-sequence-length must be positive")
    try:
        return extract_latents(args)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        _error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
