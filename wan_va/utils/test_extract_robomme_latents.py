from __future__ import annotations

import contextlib
import importlib.util
import inspect
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path


def _load_extractor():
    path = Path(__file__).with_name("extract_robomme_latents.py")
    spec = importlib.util.spec_from_file_location("robomme_latent_extractor", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


extractor = _load_extractor()


class RoboMMELatentExtractorTest(unittest.TestCase):
    def test_camera_encoding_uses_public_wan_vae_api(self):
        source = inspect.getsource(extractor._encode_camera_segment)
        self.assertIn("bundle.vae.encode(video)", source)
        self.assertNotIn("streaming_vae", source)

    def test_build_segment_jobs_and_output_path(self):
        jobs = extractor.build_segment_jobs(
            [{
                "episode_index": 1001,
                "length": 20,
                "tasks": ["move cube"],
                "action_config": [{
                    "start_frame": 2,
                    "end_frame": 18,
                    "action_text": "move cube",
                }],
            }],
            1000,
        )
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].episode_chunk, 1)
        path = extractor._latent_path(Path("dataset"), "image", jobs[0])
        self.assertEqual(
            path.as_posix(),
            "dataset/latents/chunk-001/image/episode_001001_2_18.pth",
        )

    def test_missing_action_config_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "action_config"):
            extractor.build_segment_jobs(
                [{"episode_index": 0, "length": 10, "tasks": ["task"]}],
                1000,
            )

    def test_frame_selection_preserves_integer_stride(self):
        ids, stride, fps = extractor.select_frame_ids(0, 10, 10.0, None)
        self.assertEqual(ids, list(range(10)))
        self.assertEqual(stride, 1)
        self.assertEqual(fps, 10.0)

        ids, stride, fps = extractor.select_frame_ids(2, 20, 30.0, 10.0)
        self.assertEqual(ids, [2, 5, 8, 11, 14, 17])
        self.assertEqual(stride, 3)
        self.assertEqual(fps, 10.0)

        with self.assertRaisesRegex(ValueError, "positive integer"):
            extractor.select_frame_ids(0, 20, 15.0, 10.0)

    def test_segment_payload_matches_legacy_loader_contract(self):
        job = extractor.SegmentJob(
            episode_index=7,
            episode_chunk=0,
            episode_length=12,
            start_frame=0,
            end_frame=12,
            action_text="move cube",
        )
        payload = extractor._segment_payload(
            latent="latent-tensor",
            latent_frames=3,
            latent_height=32,
            latent_width=32,
            text_embedding="text-tensor",
            job=job,
            frame_ids=list(range(12)),
            target_fps=10.0,
            source_fps=10.0,
            height=256,
            width=256,
        )
        self.assertEqual(
            set(payload),
            {
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
            },
        )
        self.assertEqual(payload["latent_num_frames"], 3)
        self.assertEqual(payload["video_num_frames"], 12)
        self.assertEqual(payload["text"], "move cube")
        self.assertEqual(payload["fps"], 10)
        self.assertEqual(payload["ori_fps"], 10)

    def test_dry_run_needs_no_torch_or_parquet(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "meta").mkdir()
            info = {
                "codebase_version": "v2.1",
                "fps": 10,
                "chunks_size": 1000,
                "data_path": (
                    "data/chunk-{episode_chunk:03d}/"
                    "episode_{episode_index:06d}.parquet"
                ),
                "features": {
                    "image": {"dtype": "image", "shape": [256, 256, 3]},
                    "wrist_image": {
                        "dtype": "image",
                        "shape": [256, 256, 3],
                    },
                },
            }
            (root / "meta" / "info.json").write_text(
                json.dumps(info), encoding="utf-8"
            )
            episode = {
                "episode_index": 0,
                "length": 8,
                "tasks": ["move cube"],
                "action_config": [{
                    "start_frame": 0,
                    "end_frame": 8,
                    "action_text": "move cube",
                }],
            }
            (root / "meta" / "episodes.jsonl").write_text(
                json.dumps(episode) + "\n", encoding="utf-8"
            )

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                status = extractor.main([
                    "--dataset-root",
                    str(root),
                    "--dry-run",
                ])
            self.assertEqual(status, 0)
            self.assertIn("1 segment(s)", output.getvalue())
            self.assertIn("2 latent file(s) pending", output.getvalue())


if __name__ == "__main__":
    unittest.main()
