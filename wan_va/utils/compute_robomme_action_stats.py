"""从官方 RoboMME LeRobot action 计算 LingBot 分位数。

该工具只读取 Parquet 的 ``actions`` 列，不会解码图像。生成的 JSON 通过
``LINGBOT_ROBOMME_ACTION_STATS=/path/to/robomme_action_stats.json`` 加载。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _read_action_batches(dataset_path: Path, action_key: str):
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError(
            'pyarrow is required to read the official RoboMME LeRobot '
            'Parquet files; install the LingBot post-training dependencies'
        ) from error

    parquet_files = sorted((dataset_path / 'data').glob('chunk-*/*.parquet'))
    if not parquet_files:
        raise FileNotFoundError(
            f'No LeRobot parquet files found under {dataset_path / "data"}')

    for parquet_path in parquet_files:
        parquet = pq.ParquetFile(parquet_path)
        for batch in parquet.iter_batches(columns=[action_key], batch_size=65536):
            actions = np.asarray(batch.column(0).to_pylist(), dtype=np.float32)
            if actions.ndim != 2 or actions.shape[1] != 8:
                raise ValueError(
                    f'{parquet_path} column {action_key!r} must be (*, 8), '
                    f'got {actions.shape}')
            if not np.isfinite(actions).all():
                raise ValueError(
                    f'{parquet_path} column {action_key!r} contains NaN or Inf')
            yield actions


def compute_stats(dataset_path: Path, action_key: str = 'actions') -> dict:
    action_batches = list(_read_action_batches(dataset_path, action_key))
    actions = np.concatenate(action_batches, axis=0)
    return {
        'action_representation': 'absolute_joint',
        'action_key': action_key,
        'num_actions': int(actions.shape[0]),
        'q01': np.quantile(actions, 0.01, axis=0).tolist(),
        'q99': np.quantile(actions, 0.99, axis=0).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description='计算 RoboMME 绝对 8D action 的 q01/q99')
    parser.add_argument('dataset_path', type=Path, help='LeRobot 数据集根目录')
    parser.add_argument('--action-key', default='actions', help='action 列名')
    parser.add_argument(
        '--output', type=Path, default=Path('robomme_action_stats.json'),
        help='输出 JSON 路径')
    args = parser.parse_args()

    stats = compute_stats(args.dataset_path, args.action_key)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(stats, indent=2) + '\n', encoding='utf-8')
    print(f'Wrote {stats["num_actions"]} actions to {args.output}')


if __name__ == '__main__':
    main()
