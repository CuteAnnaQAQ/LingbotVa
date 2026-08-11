from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


def _load_sibling(module_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(
        module_name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    # dataclasses 会通过 sys.modules 解析延迟求值的类型注解。
    import sys
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_validator = _load_sibling(
    "robomme_data_validator", "validate_robomme_lerobot.py")
_action_config = _load_sibling(
    "robomme_action_config", "add_robomme_action_config.py")
audit_dataset = _validator.audit_dataset
add_whole_episode_segments = _action_config.add_whole_episode_segments


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class RoboMMEDataCompatibilityTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.info = {
            "codebase_version": "v2.1",
            "robot_type": "panda",
            "total_episodes": 1,
            "total_frames": 2,
            "total_tasks": 1,
            "total_videos": 0,
            "total_chunks": 1,
            "chunks_size": 1000,
            "fps": 10,
            "splits": {"train": "0:1"},
            "data_path": (
                "data/chunk-{episode_chunk:03d}/"
                "episode_{episode_index:06d}.parquet"),
            "video_path": (
                "videos/chunk-{episode_chunk:03d}/{video_key}/"
                "episode_{episode_index:06d}.mp4"),
            "features": {
                "image": {"dtype": "image", "shape": [256, 256, 3]},
                "wrist_image": {"dtype": "image", "shape": [256, 256, 3]},
                "state": {"dtype": "float32", "shape": [8]},
                "actions": {"dtype": "float32", "shape": [8]},
                "timestamp": {"dtype": "float32", "shape": [1]},
                "frame_index": {"dtype": "int64", "shape": [1]},
                "episode_index": {"dtype": "int64", "shape": [1]},
                "index": {"dtype": "int64", "shape": [1]},
                "task_index": {"dtype": "int64", "shape": [1]},
            },
        }
        self.episode = {
            "episode_index": 0,
            "tasks": ["move the cube"],
            "length": 2,
        }
        _write_json(self.root / "meta" / "info.json", self.info)
        _write_jsonl(self.root / "meta" / "episodes.jsonl", [self.episode])
        _write_jsonl(
            self.root / "meta" / "tasks.jsonl",
            [{"task_index": 0, "task": "move the cube"}],
        )
        _write_jsonl(
            self.root / "meta" / "episodes_stats.jsonl",
            [{"episode_index": 0, "stats": {}}],
        )
        data_path = self.root / "data" / "chunk-000" / "episode_000000.parquet"
        data_path.parent.mkdir(parents=True)
        data_path.touch()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_raw_image_dataset_does_not_require_videos(self):
        result = audit_dataset(self.root)
        self.assertTrue(result.raw_compatible, result.raw_errors)
        self.assertFalse(result.lingbot_ready)
        self.assertTrue(any("videos/ is not required" in note for note in result.notes))
        self.assertTrue(any("action_config" in error for error in result.ready_errors))

    def test_whole_episode_config_and_lingbot_artifacts_pass(self):
        episodes = add_whole_episode_segments([self.episode])
        self.assertEqual(
            episodes[0]["action_config"],
            [{
                "start_frame": 0,
                "end_frame": 2,
                "action_text": "move the cube",
            }],
        )
        _write_jsonl(self.root / "meta" / "episodes.jsonl", episodes)
        (self.root / "empty_emb.pt").touch()
        for camera in ("image", "wrist_image"):
            latent = (
                self.root / "latents" / "chunk-000" / camera /
                "episode_000000_0_2.pth")
            latent.parent.mkdir(parents=True, exist_ok=True)
            latent.touch()
        stats_path = self.root / "action_stats.json"
        _write_json(stats_path, {
            "action_representation": "absolute_joint",
            "q01": [-1.0] * 8,
            "q99": [1.0] * 8,
        })

        result = audit_dataset(self.root, action_stats=stats_path)
        self.assertTrue(result.raw_compatible, result.raw_errors)
        self.assertTrue(result.lingbot_ready, result.ready_errors)


if __name__ == "__main__":
    unittest.main()
