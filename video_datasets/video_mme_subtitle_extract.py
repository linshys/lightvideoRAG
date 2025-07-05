import os
import cv2
import numpy as np
import pysubs2
from PIL import Image
from moviepy.video.io.VideoFileClip import VideoFileClip


def get_seq_frames(total_num_frames, desired_num_frames):
    seg_size = float(total_num_frames - 1) / desired_num_frames
    seq = []
    for i in range(desired_num_frames):
        start = int(np.round(seg_size * i))
        end = int(np.round(seg_size * (i + 1)))
        seq.append((start + end) // 2)
    return seq


def slice_frames(video_path, srt_path, num_frames):
    """
    Extract frames from video and subtitles from srt.
    Return frames as PIL.Image list and subtitles as string.

    Returns:
        frames (List[PIL.Image])
        subtitles (str)
    """
    print(f"Extracting from video: {video_path}")

    frames = []
    sampled_indices = []
    with VideoFileClip(video_path) as clip:
        duration = int(clip.duration)
        fps = clip.fps

        total_frames = int(clip.fps * clip.duration)
        if total_frames <= num_frames:
            sampled_indices = range(total_frames)
        else:
            stride = total_frames / num_frames
            sampled_indices = [min(total_frames - 1, int((stride * i + stride * (i + 1)) / 2)) for i in range(num_frames)]
        frames = [clip.get_frame(index / clip.fps) for index in sampled_indices]
        frames = [Image.fromarray(frame, mode='RGB') for frame in frames]

    if srt_path and os.path.exists(srt_path):
        subs = pysubs2.load(srt_path, encoding="utf-8")
        subtitles = []
        for frame_id in sampled_indices:
            cur_time = pysubs2.make_time(fps=fps, frames=frame_id)
            sub_text = ""
            for sub in subs:
                if sub.start < cur_time < sub.end:
                    sub_text = sub.text.replace("\\N", " ")
                    break
            if sub_text.strip():
                subtitles.append(sub_text)
        subtitles = "\n".join(subtitles)
    else:
        subtitles = ""

    return frames, subtitles

def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Extract frames from a video and corresponding subtitles.")
    parser.add_argument("--video_path", type=str, help="Path to the video file.")
    parser.add_argument("--srt_path", type=str, default=None, help="Path to the subtitles file.")
    parser.add_argument("--num_frames", type=int, default=2, help="Number of frames to extract.")
    parser.add_argument("--output_path", type=str, default="output", help="Path to the output directory.")


    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    slice_frames(args.video_path, args.srt_path, args.num_frames)