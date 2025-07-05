import os
import tempfile
import subprocess
import logging
import torch
from moviepy.video.io.VideoFileClip import VideoFileClip
from tqdm import tqdm
from faster_whisper import WhisperModel
from .._utils import logger
import datetime

def speech_to_text(video_name, working_dir, segment_index2name, audio_output_format):
    model = WhisperModel("./faster-distil-whisper-large-v3")
    model.logger.setLevel(logging.WARNING)
    
    cache_path = os.path.join(working_dir, '_cache', video_name)
    
    transcripts = {}
    for index in tqdm(segment_index2name, desc=f"Speech Recognition {video_name}"):
        segment_name = segment_index2name[index]
        audio_file = os.path.join(cache_path, f"{segment_name}.{audio_output_format}")
        segments, info = model.transcribe(audio_file)
        result = ""
        for segment in segments:
            result += "[%.2fs -> %.2fs] %s\n" % (segment.start, segment.end, segment.text)
        transcripts[index] = result
    
    return transcripts


def extract_transcripts(video_path, audio_output_format):
    model = WhisperModel("./faster-distil-whisper-large-v3")
    model.logger.setLevel(logging.WARNING)

    video_name = os.path.basename(video_path).rsplit('.', 1)[0]
    video = VideoFileClip(video_path)
    audio = video.audio
    if audio is None:
        print(f"No audio track for video: {video_name}")
        return []
    audio_duration = audio.duration
    video.close()

    # Step 3: 创建内存临时音频文件（更快）
    ramdisk_dir = "/dev/shm" if os.path.exists("/dev/shm") else None
    with tempfile.NamedTemporaryFile(suffix=f".{audio_output_format}", dir=ramdisk_dir) as tmp_audio:
        audio_file = tmp_audio.name

        # Step 4: 使用 ffmpeg 提取音频
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vn",
            "-acodec", "pcm_s16le",  # 对应 .wav 格式
            audio_file
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        # Step 5: 语音转录
        try:
            transcripts = []
            segments, _ = model.transcribe(
                audio_file,
                vad_filter=True,
                word_timestamps=True
            )
            with tqdm(total=audio_duration, desc="Transcribing", unit="sec") as pbar:
                for segment in segments:
                    transcripts.append({
                        "start_stamp": round(segment.start, 3),
                        "end_stamp": round(segment.end, 3),
                        "text": segment.text
                    })
                    pbar.n = segment.end
                    pbar.refresh()

            return transcripts
        except Exception as e:
            print(f"[ERROR] Transcription failed for {video_path}: {e}")
            return []
        finally:
            del model
            torch.cuda.empty_cache()

def convert_to_text_with_timestamp(audio_path,start_time, max_len):
    model = WhisperModel("./faster-distil-whisper-large-v3")
    model.logger.setLevel(logging.WARNING)
    try:
        segments, _ = model.transcribe(
            audio_path, vad_filter=True, word_timestamps=True)
        if not segments:
            logger.error(f"⚠️ Whisper 没有检测到语音")
            return ""
    except Exception as e:
        logger.error(f"Whisper 处理失败: {e}")
        return ""
    transcripts = ""
    for segment in segments:
        transcripts += "[%s -> %s] %s" % (
            seconds_to_hms(segment.start + start_time),
            seconds_to_hms(segment.end + start_time),
            segment.text
        )
    if max_len != -1 and len(transcripts) > max_len:
        transcripts = transcripts[:max_len]

    return transcripts

def seconds_to_hms(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))
