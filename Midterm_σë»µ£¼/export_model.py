#!/usr/bin/env python3
"""
导出 HuggingFace 缓存中的 Qwen3.5-2B 模型到指定目录。
将 snapshots 中的符号链接解析为实际文件并复制出来。
"""

import os
import shutil
import argparse
from pathlib import Path


CACHE_DIR = Path("/Users/billxu/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B")


def find_snapshot_dir(cache_dir: Path) -> Path:
    """找到最新的 snapshot 目录（优先使用 refs/main 指向的版本）."""
    refs_main = cache_dir / "refs" / "main"
    if refs_main.exists():
        commit_hash = refs_main.read_text().strip()
        snapshot_dir = cache_dir / "snapshots" / commit_hash
        if snapshot_dir.exists():
            return snapshot_dir

    snapshots_dir = cache_dir / "snapshots"
    snapshots = sorted(snapshots_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    if not snapshots:
        raise FileNotFoundError(f"未找到任何 snapshot: {snapshots_dir}")
    return snapshots[0]


def export_model(output_dir: Path, snapshot_dir: Path, verbose: bool = True):
    """将 snapshot 中的所有文件（解析符号链接）复制到 output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    files = list(snapshot_dir.iterdir())
    # 过滤隐藏文件（如 .gitattributes）
    files = [f for f in files if not f.name.startswith(".")]

    total = len(files)
    for i, src in enumerate(files, 1):
        # 解析符号链接得到真实路径
        real_src = src.resolve()
        dst = output_dir / src.name

        size_mb = real_src.stat().st_size / (1024 ** 2)
        if verbose:
            print(f"[{i}/{total}] 复制 {src.name}  ({size_mb:.1f} MB) ...")

        shutil.copy2(real_src, dst)

    if verbose:
        print(f"\n导出完成！共 {total} 个文件 -> {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="导出 HuggingFace 缓存中的 Qwen3.5-2B 模型")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=str(Path.home() / "Desktop" / "Qwen3.5-2B"),
        help="导出目标目录（默认: ~/Desktop/Qwen3.5-2B）",
    )
    parser.add_argument(
        "--cache",
        type=str,
        default=str(CACHE_DIR),
        help=f"HuggingFace 缓存目录（默认: {CACHE_DIR}）",
    )
    parser.add_argument("--quiet", "-q", action="store_true", help="静默模式，不打印每个文件")
    args = parser.parse_args()

    cache_dir = Path(args.cache)
    output_dir = Path(args.output)

    if not cache_dir.exists():
        raise SystemExit(f"错误：缓存目录不存在: {cache_dir}")

    snapshot_dir = find_snapshot_dir(cache_dir)
    print(f"使用 snapshot: {snapshot_dir.name}")
    print(f"导出到: {output_dir}\n")

    export_model(output_dir, snapshot_dir, verbose=not args.quiet)


if __name__ == "__main__":
    main()
