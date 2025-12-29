#!/bin/bash
# ffmpeg 提取音频和帧
ffmpeg -i "$1" "$2.wav"
ffmpeg -i "$1" "$2_frames/frame_%04d.jpg" -vf fps=1
