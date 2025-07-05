import json
import subprocess
import tempfile
import datetime

import pysubs2

from .base import (
    QueryParam,
)

from ._utils import logger

from ._vlm import (
    ovis2_refine_retrieval_query,
    ovis2_inference
)
from ._videoutil import (
    filter_similar_frames,
    filter_similar_frames_kornia_gpu,
    convert_to_text_with_timestamp
)

from collections import defaultdict
from moviepy.editor import VideoFileClip
import os
import torch
import shutil
from PIL import Image

async def lightvideorag_query(
        query,
        video_path_db,
        video_frames_feature_vdb,
        video_transcript_feature_vdb,
        transcripts_meta_kv,
        video_frames_transcript_vdb,
        query_param: QueryParam,
        global_config: dict,
) -> str:
    # Step1: convert query for both vision and text retrieve
    refined_prompt = ovis2_refine_retrieval_query(query)

    ## TODO topk筛选条件
    # Step2: retrieve
    retrieve_frames = await video_frames_feature_vdb.query(
        refined_prompt['visual_retrieval'],
        top_k = query_param.top_k,
        threshold = -1)

    retrieve_transcript =  await video_transcript_feature_vdb.query(
        refined_prompt['transcript_retrieval'],
        top_k=query_param.top_k,
        threshold = -1)

    # Step3: postProcess
    grouped_video_dict =  select_and_group_items(retrieve_frames,retrieve_transcript, frame_source=3, transcript_source=2)
    grouped_video_dict_with_extra = sampling_extra_frames(grouped_video_dict,video_path_db, num_frames=10)
    image_transcript_pairs = get_image_transcript_pairs(grouped_video_dict_with_extra, video_path_db, global_config,query_param.extra_for_rag)

    # Step4: inference
    response = ovis2_inference(query, frames = image_transcript_pairs['vision'], transcripts= image_transcript_pairs['text'])

    # Step5： clean cache
    video_segment_cache_path = os.path.join(global_config['working_dir'], '_cache')
    if os.path.exists(video_segment_cache_path):
        shutil.rmtree(video_segment_cache_path)

    return response


def select_and_group_items(frames, transcripts, frame_source=3, transcript_source=3):
    total = frame_source + transcript_source

    # 初始采样
    top_frames = frames[:frame_source]
    top_transcripts = transcripts[:transcript_source]
    merged = top_frames + top_transcripts

    needed = total - len(merged)

    # 如果有缺，尝试从剩下的 frames 和 transcripts 中补
    if needed > 0:
        extra_frames = frames[frame_source:]
        extra_transcripts = transcripts[transcript_source:]

        # 优先补 frames
        supplement = extra_frames[:needed]
        still_needed = needed - len(supplement)

        # 再补 transcripts
        if still_needed > 0:
            supplement += extra_transcripts[:still_needed]

        merged += supplement

    # 构建分组字典
    grouped = defaultdict(list)
    for item in merged[:total]:  # 确保不超 total
        video_name = item["__video_name__"]
        item["__type__"] = "retrieval"
        if "__start_stamp__" in item and "__end_stamp__" in item:
            item["__timestamp__"] = round((item["__start_stamp__"] + item["__end_stamp__"]) / 2, 2)
        grouped[video_name].append(item)

    # 按时间排序
    for video_name in grouped:
        grouped[video_name].sort(key=lambda x: x.get("__timestamp__", 0))

    return grouped

def sampling_extra_frames(grouped_video_dict,video_path_db, num_frames = 10):
    for video_name, items in grouped_video_dict.items():
        video_path = video_path_db._data.get(video_name)
        if not video_path:
            logger.info(f"Can't find video path for {video_name}, skip retrieve")
            continue

        with VideoFileClip(video_path) as clip:
            total_frames = int(clip.fps * clip.duration)
            if total_frames <= num_frames:
                sampled_indices = list(range(total_frames))
            else:
                stride = (total_frames - 1) / (num_frames - 1)
                sampled_indices = [
                    min(total_frames - 1, int(round(stride * i)))
                    for i in range(num_frames)
                ]

            for index in sampled_indices:
                t = min(index / clip.fps, clip.duration - 1) # 1s buffer
                try:
                    frame_array = clip.get_frame(t)
                    frame_img = Image.fromarray(frame_array, mode='RGB')
                    timestamp = round(t, 2)

                    items.append({
                        '__type__': 'uniform_sampling',
                        'raw_frame': frame_img,
                        '__timestamp__': timestamp,
                    })
                except Exception as e:
                    logger.warning(f"Failed to read frame at t={t}s for {video_name}: {e}")
                    continue
    # 按时间排序
    for video_name in grouped_video_dict:
        grouped_video_dict[video_name].sort(key=lambda x: x.get("__timestamp__", 0))
    return grouped_video_dict

def get_image_transcript_pairs(video_data_dict, video_path_db, global_config, extra:dict = None):
    image_transcript_pairs = defaultdict(list)
    for video_name, items in video_data_dict.items():
        video_path = None
        if video_name in video_path_db._data:
            video_path =  video_path_db._data[video_name]
            if video_path is None:
                logger.info(f"can't find video path for {video_name}, skip retrieve")
                continue

        clip = None
        try:
            clip = VideoFileClip(video_path)
            for item in items:
                if item['__type__'] == 'uniform_sampling':
                    frame= {
                        item['__timestamp__']:item['raw_frame']
                    }
                    text = ''   #TODO 是否加文字
                    image_transcript_pairs['vision'].append(frame)
                    image_transcript_pairs['text'].append(text)
                    continue

                timestamp = item['__timestamp__']
                frame = extract_frame(clip, timestamp)
                if extra is not None and extra.get('subtitle_path') is not None:
                    text = insert_subtitles(timestamp, duration=30, extra = extra)
                else:
                    text = extract_audio(clip,timestamp,global_config=global_config,duration=30)
                image_transcript_pairs['vision'].append(frame)
                image_transcript_pairs['text'].append(text)
        except Exception as e:
            logger.error(f"process {video_name} error happened: {e}")
        finally:
            if clip is not None:
                clip.close()

    torch.cuda.empty_cache()
    return image_transcript_pairs

def extract_frame(video: VideoFileClip, timestamp,frame_interval=1 ):
    if timestamp > video.duration:
        print(f"⚠️  warning, time stamp {timestamp} extends video duration {video.duration}s，skip")
        return []

    frames = {}
    timestamps = [timestamp - 2 * frame_interval, timestamp - frame_interval, timestamp,
                  timestamp + frame_interval, timestamp + 2 * frame_interval]

    for t in timestamps:
        if 0 <= t <= video.duration:
            frame = video.get_frame(t)
            frames[t] = frame

    return filter_similar_frames_kornia_gpu(list(frames.values()), list(frames.keys()), [timestamp], ssim_threshold=0.9)

def extract_audio(video: VideoFileClip, timestamp, global_config, duration=30):
    if video.audio is None:
        print("⚠️ No audio track, skip")
        return ""

    # Step 1: 计算截取区间（中心 ± duration / 2）
    half = duration / 2
    start_time = max(0, timestamp - half)
    end_time = min(video.duration, timestamp + half)

    # Step 2: 设置临时文件路径（/dev/shm 优先）
    ramdisk_dir = "/dev/shm" if os.path.exists("/dev/shm") else None
    output_format = global_config.get("audio_output_format", "wav")

    with tempfile.NamedTemporaryFile(suffix=f".{output_format}", dir=ramdisk_dir, delete=True) as tmp_audio:
        audio_file = tmp_audio.name

        # Step 3: 构造 ffmpeg 命令
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_time),
            "-to", str(end_time),
            "-i", video.filename,
            "-vn",
            "-acodec", "pcm_s16le" if output_format == "wav" else "copy",
            audio_file
        ]

        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

            # Step 4: Whisper 语音识别
            return convert_to_text_with_timestamp(audio_file, start_time, max_len=-1)

        except subprocess.CalledProcessError as e:
            logger.error(f"❌ FFmpeg 提取音频失败: {e}")
            return ""

def insert_subtitles(timestamp, duration=30, extra:dict=None):
    transcripts = ""
    task_name = extra.get('task','')

    if task_name == 'mme':
        srt_path = extra.get('subtitle_path')
        if srt_path and os.path.exists(srt_path):
            subs = pysubs2.load(srt_path, encoding="utf-8")
            subtitles = []

            cur_time = timestamp * 1000
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

        return subtitles

    elif task_name == 'longvideobench':
        sub_path = extra.get('subtitle_path')
        video_duration = extra.get('duration', 0)
        start_offset = extra.get('starting_timestamp_for_subtitles', 0)

        if sub_path is None:
            print("[警告][推理中] 未提供 subtitle_path")
            return transcripts

        with open(sub_path, 'r', encoding='utf-8') as f:
            subtitles = json.load(f)

        transcripts = load_subtitles_longvideobench([timestamp], subtitles, start_offset, video_duration, max_len=256)

        # transcripts = load_subtitles_with_window([timestamp], subtitles, start_offset, video_duration,
        #                            context_window= duration, max_len=256)
    return transcripts

def load_subtitles_longvideobench(timestamps, subtitles,
                                  start_offset, video_duration,
                                  max_len=256):
    lines = []

    for subtitle in subtitles:
        if "timestamp" in subtitle:
            start, end = subtitle["timestamp"]
            if not isinstance(end, float):
                end = video_duration
            text = subtitle["text"]
        else:
            start = time_str_to_seconds(subtitle["start"])
            end = time_str_to_seconds(subtitle["end"])
            text = subtitle["line"]

        start -= start_offset
        end -= start_offset

        # 字幕过短时，扩展为 1 秒窗口
        if end - start < 1:
            center = (start + end) / 2
            start = center - 0.5
            end = center + 0.5

        # 如果字幕区间内有帧，就保留
        has_covering_frame = any(start <= ts <= end for ts in timestamps)
        if has_covering_frame:
            lines.append(
                "[%s -> %s] %s" % (
                    seconds_to_hms(start),
                    seconds_to_hms(end),
                    text.strip()
                )
            )

    transcripts = " ".join(lines)
    if len(transcripts) > max_len:
        transcripts = transcripts[:max_len]

    return transcripts

def load_subtitles_with_window(timestamps, subtitles,
                               start_offset, video_duration,
                               context_window = 30, max_len=256):
    lines = []
    half_window = context_window / 2
    end_limit = start_offset + video_duration

    for subtitle in subtitles:
        # 解析原始起止时间 & 文本
        if "timestamp" in subtitle:
            start, end = subtitle["timestamp"]
            if not isinstance(end, float):
                end = video_duration + start_offset
            text = subtitle["text"]
        else:
            start = time_str_to_seconds(subtitle["start"])
            end = time_str_to_seconds(subtitle["end"])
            text = subtitle["line"]

        if end < start_offset:
            continue

        if start > end_limit:
            break

        start -= start_offset
        end -= start_offset
        center = (start + end) / 2

        for ts in timestamps:
            if abs(center - ts) <= half_window:
                lines.append(
                    "[%s -> %s] %s" % (
                        seconds_to_hms(start),
                        seconds_to_hms(end),
                        text.strip()
                    )
                )

    transcripts = " ".join(lines)
    if len(transcripts) > max_len:
        transcripts = transcripts[:max_len]

    return transcripts


def seconds_to_hms(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))

def time_str_to_seconds(time_str):
    try:
        h, m, s = time_str.strip().split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)
    except Exception as e:
        print(f"[格式错误] 无法解析时间字符串 '{time_str}': {e}")
        return 0.0