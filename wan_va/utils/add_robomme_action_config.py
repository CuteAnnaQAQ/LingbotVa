"""为 RoboMME episodes.jsonl 增加 LingBot 整 episode action 分段。"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any


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


def add_whole_episode_segments(
    episodes: list[dict[str, Any]], *, replace: bool = False
) -> list[dict[str, Any]]:
    output = []
    for row in episodes:
        episode_index = row.get("episode_index")
        length = row.get("length")
        tasks = row.get("tasks")
        if not isinstance(length, int) or length <= 0:
            raise ValueError(
                f"episode {episode_index}: length must be a positive integer")
        if not (isinstance(tasks, list) and tasks
                and isinstance(tasks[0], str) and tasks[0].strip()):
            raise ValueError(
                f"episode {episode_index}: tasks[0] must be a language instruction")
        updated = dict(row)
        if replace or not updated.get("action_config"):
            updated["action_config"] = [{
                "start_frame": 0,
                "end_frame": length,
                "action_text": tasks[0].strip(),
            }]
        output.append(updated)
    return output


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "把 RoboMME episodes.jsonl 中每条 tasks[0] 映射为一个整 episode "
            "LingBot action_config 分段"))
    parser.add_argument("--dataset-root", required=True, type=Path)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument(
        "--output", type=Path,
        help="写入新的 JSONL 文件，不修改原数据集")
    destination.add_argument(
        "--in-place", action="store_true",
        help="创建带时间戳的备份后替换 meta/episodes.jsonl")
    parser.add_argument(
        "--replace", action="store_true",
        help="替换已有 action_config，而不是保留它")
    args = parser.parse_args()

    source = args.dataset_root / "meta" / "episodes.jsonl"
    if not source.is_file():
        raise FileNotFoundError(f"episodes metadata not found: {source}")
    rows = add_whole_episode_segments(_read_jsonl(source), replace=args.replace)

    if args.in_place:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = source.with_name(f"episodes.jsonl.before-lingbot-{stamp}.bak")
        shutil.copy2(source, backup)
        target = source
        print(f"Backup: {backup}")
    else:
        target = args.output
        if target.resolve() == source.resolve():
            raise ValueError("Use --in-place when replacing meta/episodes.jsonl")

    _atomic_write_jsonl(target, rows)
    print(f"Wrote {len(rows)} episodes to {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
