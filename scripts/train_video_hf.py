#!/usr/bin/env python3
r"""视频预处理小脚本（独立运行）

功能：
- 使用 ffmpeg 提取音频为 16k 单声道 WAV
- 使用 ffmpeg 按指定 FPS 导出帧到目录
- 可选裁剪时间段、缩放帧大小
- 输出包含生成文件路径与基本统计的 JSON 到 stdout 或指定文件

用途（示例）：
python scripts/preprocess_video.py input.mp4 --out-dir temp/job1 --fps 5 --samplerate 16000

注意：依赖系统安装的 ffmpeg（需在 PATH 或通过 FFMPEG_PATH 指定）。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple


def find_ffmpeg() -> Optional[str]:
    p = os.environ.get('FFMPEG_PATH')
    if p:
        # 如果传入目录，尝试拼接可执行
        if os.path.isdir(p):
            exe = os.path.join(p, 'ffmpeg.exe' if os.name == 'nt' else 'ffmpeg')
            if os.path.exists(exe):
                return exe
        if os.path.exists(p):
            return p
    exe = shutil.which('ffmpeg')
    return exe


def run_cmd(cmd: list[str]) -> Tuple[int, str, str]:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = proc.communicate()
    return proc.returncode, out, err


def extract_audio(ffmpeg: str, input_path: str, out_wav: str, samplerate: int = 16000, start: Optional[float] = None, end: Optional[float] = None) -> None:
    cmd = [ffmpeg, '-y']
    if start is not None:
        cmd += ['-ss', str(start)]
    cmd += ['-i', input_path]
    if end is not None:
        # duration = end - start (if start provided)
        if start is not None:
            duration = float(end) - float(start)
            if duration > 0:
                cmd += ['-t', str(duration)]
        else:
            cmd += ['-to', str(end)]
    cmd += ['-vn', '-ac', '1', '-ar', str(samplerate), out_wav]
    code, out, err = run_cmd(cmd)
    if code != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed: {err.strip()}")


def extract_frames(ffmpeg: str, input_path: str, out_dir: str, fps: int = 5, width: Optional[int] = None, height: Optional[int] = None, start: Optional[float] = None, end: Optional[float] = None) -> None:
    os.makedirs(out_dir, exist_ok=True)
    pattern = os.path.join(out_dir, 'frame_%04d.jpg')
    vf = f"fps={fps}"
    if width or height:
        w = width or -1
        h = height or -1
        # ffmpeg scale: -1 保持纵横比
        vf += f",scale={w}:{h}"
    cmd = [ffmpeg, '-y']
    if start is not None:
        cmd += ['-ss', str(start)]
    cmd += ['-i', input_path]
    if end is not None:
        if start is not None:
            duration = float(end) - float(start)
            if duration > 0:
                cmd += ['-t', str(duration)]
        else:
            cmd += ['-to', str(end)]
    cmd += ['-vf', vf, pattern]
    code, out, err = run_cmd(cmd)
    if code != 0:
        raise RuntimeError(f"ffmpeg frame extraction failed: {err.strip()}")


def count_frames(frames_dir: str) -> int:
    try:
        files = [f for f in os.listdir(frames_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        return len(files)
    except Exception:
        return 0


def parse_args():
    p = argparse.ArgumentParser(description='Video preprocessing: extract audio (16k mono) and frames via ffmpeg')
    p.add_argument('input', help='input video file')
    p.add_argument('--out-dir', default=None, help='output base dir (default: ./tmp_preprocess/<input_basename>)')
    p.add_argument('--fps', type=int, default=5, help='frames per second to extract')
    p.add_argument('--samplerate', type=int, default=16000, help='audio sample rate (Hz)')
    p.add_argument('--width', type=int, default=None, help='optional frame width (scale)')
    p.add_argument('--height', type=int, default=None, help='optional frame height (scale)')
    p.add_argument('--start', type=float, default=None, help='start time in seconds')
    p.add_argument('--end', type=float, default=None, help='end time in seconds')
    p.add_argument('--metadata-out', default=None, help='write metadata JSON to this file instead of stdout')
    return p.parse_args()


def main():
    args = parse_args()
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        print('ffmpeg not found. Install ffmpeg or set FFMPEG_PATH.', file=sys.stderr)
        sys.exit(2)

    inp = os.path.abspath(args.input)
    if not os.path.exists(inp):
        print(f'input not found: {inp}', file=sys.stderr)
        sys.exit(2)

    base = args.out_dir or os.path.join('tmp_preprocess', Path(inp).stem)
    base = os.path.abspath(base)
    os.makedirs(base, exist_ok=True)

    audio_path = os.path.join(base, Path(inp).stem + '.wav')
    frames_dir = os.path.join(base, 'frames')

    try:
        extract_audio(ffmpeg, inp, audio_path, samplerate=args.samplerate, start=args.start, end=args.end)
    except Exception as e:
        print(f'Error extracting audio: {e}', file=sys.stderr)
        sys.exit(3)

    try:
        extract_frames(ffmpeg, inp, frames_dir, fps=args.fps, width=args.width, height=args.height, start=args.start, end=args.end)
    except Exception as e:
        print(f'Error extracting frames: {e}', file=sys.stderr)
        sys.exit(4)

    n_frames = count_frames(frames_dir)
    metadata = {
        'input': inp,
        'out_dir': base,
        'audio': audio_path,
        'frames_dir': frames_dir,
        'n_frames': n_frames,
        'fps': args.fps,
        'samplerate': args.samplerate,
        'start': args.start,
        'end': args.end,
    }

    out_json = json.dumps(metadata, ensure_ascii=False, indent=2)
    if args.metadata_out:
        with open(args.metadata_out, 'w', encoding='utf-8') as f:
            f.write(out_json)
        print(f'Wrote metadata to {args.metadata_out}')
    else:
        print(out_json)


if __name__ == '__main__':
    main()
