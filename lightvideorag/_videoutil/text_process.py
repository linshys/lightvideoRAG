import json
from .asr import extract_transcripts

def load_transcripts(video_path, audio_output_format, extra:dict):
    transcripts = []
    if extra is not None and extra.get('subtitle_path') is not None:
        transcripts = load_subtitles(extra)
    else:
        transcripts = extract_transcripts(video_path, audio_output_format)

    return merge_transcripts(transcripts, threshold=0.05)

def load_subtitles(extra: dict):
    transcripts = []

    sub_path = extra.get('subtitle_path')
    duration = extra.get('duration', 0)
    start_offset = extra.get('starting_timestamp_for_subtitles', 0)
    end_limit = start_offset + duration

    if sub_path is None:
        print("[警告] 未提供 subtitle_path")
        return transcripts

    with open(sub_path, 'r', encoding='utf-8') as f:
        subtitles = json.load(f)

    for subtitle in subtitles:
        if "timestamp" in subtitle:
            start_stamp, end_stamp = subtitle["timestamp"]
            if not isinstance(end_stamp, float):
                end_stamp = duration+start_offset
            text = subtitle["text"]
        else:
            start_stamp = time_str_to_seconds(subtitle["start"])
            end_stamp = time_str_to_seconds(subtitle["end"])
            text = subtitle["line"]

        if end_stamp < start_offset:
            continue

        if start_stamp > end_limit:
            break

        relative_start = round(max(0.0, start_stamp - start_offset), 3)
        relative_end = round(min(duration, end_stamp - start_offset), 3)

        # 过滤极端情况：end < start（可能裁剪后导致）
        if relative_end <= relative_start:
            continue

        transcripts.append({
            "start_stamp": relative_start,
            "end_stamp": relative_end,
            "text": text
        })

    return transcripts

def time_str_to_seconds(time_str):
    try:
        h, m, s = time_str.strip().split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)
    except Exception as e:
        print(f"[格式错误] 无法解析时间字符串 '{time_str}': {e}")
        return 0.0


def merge_transcripts(transcripts, threshold=0.05):
    merged_transcripts = []
    current_index = 0
    current_segment = None

    for idx, segment in enumerate(transcripts):
        if current_segment is None:
            current_segment = segment
        else:
            # 计算两个句子之间的时间间隔
            gap = segment["start_stamp"] - current_segment["end_stamp"]

            if gap < threshold:  # 如果间隔小于 `threshold` 秒，则合并
                current_segment["text"] += " " + segment["text"]
                current_segment["end_stamp"] = segment["end_stamp"]  # 更新结束时间
            else:
                merged_transcripts.append(current_segment)
                current_index += 1
                current_segment = segment

    if current_segment:
        merged_transcripts.append(current_segment)
    ## TODO 按时间长短合并，冗余保存
    return merged_transcripts