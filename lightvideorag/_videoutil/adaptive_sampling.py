import torch
import cv2
import decord
from decord import VideoReader, gpu
import kornia
import kornia.augmentation as K
import numpy as np
from kornia.constants import Resample
from moviepy.editor import VideoFileClip
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

fps_sampling_rate = 1  # 采样频率（每秒1帧）
ssim_threshold = 0.8  # SSIM 阈值（大于此值判定为相似）
segment_duration = 600  # 每个分片的长度（秒）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_transform = transforms.Compose(
    [
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ]
)

def similarity_sampling(video_path):
    # 读取视频总时长
    overall_clip = VideoFileClip(video_path)
    duration = overall_clip.duration  # 视频总时长（秒）
    num_segments = int(np.ceil(duration / segment_duration))  # 计算分片数量

    overall_clip.reader.close()

    # 计算总帧数
    total_frames = int(duration * fps_sampling_rate)

    # 使用线程池并行处理视频片段
    with tqdm(total=total_frames, desc="Processing Video", unit="frame") as progress_bar:
        with ThreadPoolExecutor() as executor:
            futures = {}
            for i in range(num_segments):
                start_time = i * segment_duration
                end_time = min((i + 1) * segment_duration, duration)
                futures[executor.submit(process_segment,video_path, start_time, end_time, i, progress_bar)] = i

            # 收集结果，按顺序合并
            processed_tensors = []
            frame_timestamps = []  # 存储时间戳
            for future in sorted(futures.keys(), key=lambda f: futures[f]):
                segment_index, tensors = future.result()
                for timestamp, tensor in tensors:
                    processed_tensors.append(tensor)
                    frame_timestamps.append(timestamp)

    # 堆叠张量
    if processed_tensors:
        final_tensor = torch.stack(processed_tensors, dim=0)
        print(f"最终张量大小: {final_tensor.shape}")  # (batch_size, 3, 224, 224)
    else:
        final_tensor = None
        print("未保存任何帧！")

    return frame_timestamps, final_tensor

def process_segment(video_path, start_time, end_time, segment_index, progress_bar):
    """处理视频片段，并返回已处理的张量。"""
    clip = VideoFileClip(video_path).subclip(start_time, end_time)
    frame_times = np.arange(0, end_time - start_time, 1 / fps_sampling_rate)  # 计算采样时间点

    prev_frame = None
    segment_tensors = []

    for t in frame_times:
        frame = clip.get_frame(t)  # 获取时间点 t 处的帧
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # 转换为灰度

        if prev_frame is not None:
            similarity = ssim(prev_frame, gray_frame)
            if similarity > ssim_threshold:
                progress_bar.update(1)
                continue  # 丢弃相似帧

        # 转换为 PIL.Image 并预处理
        pil_image = Image.fromarray(frame)
        processed_image = data_transform(pil_image).to(device)
        segment_tensors.append((t + start_time, processed_image))
        progress_bar.update(1)
        prev_frame = gray_frame

    clip.reader.close()
    return segment_index, segment_tensors  # 直接返回，无需排序


### for retrieve
def filter_similar_frames(frames: list, timestamps: list, keep_frames: list, ssim_threshold=0.8):
    prev_frame = None
    filtered_frames = {}

    for idx, frame in enumerate(frames):
        timestamp = timestamps[idx]
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # 转换为灰度

        # 强制保留 `keep_frames` 里的帧，无论相似度如何
        if timestamp in keep_frames:
            pil_image = Image.fromarray(frame)
            filtered_frames[timestamp] = pil_image
            prev_frame = gray_frame
            continue  # 直接跳过 `SSIM` 计算

        # 🚀 计算 `SSIM`，高相似度的帧丢弃
        if prev_frame is not None:
            similarity = ssim(prev_frame, gray_frame)
            if similarity > ssim_threshold:
                continue  # 丢弃相似帧

        # 转换为 `PIL.Image` 并保存
        pil_image = Image.fromarray(frame)
        filtered_frames[timestamp] = pil_image
        prev_frame = gray_frame

    return filtered_frames

##################GPU Version####################

def __build_gpu_transform(device):
    """等价于 torchvision 的 Resize(224) + CenterCrop + Normalize"""
    return torch.nn.Sequential(
        K.Resize(size=(224, 224), resample=Resample.BICUBIC, align_corners=False),
        K.Normalize(
            mean=torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device),
            std=torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device)
        )
    )


def __get_adaptive_window_size(h, w, min_size=11, max_size=51):
    short_side = min(h, w)
    win = short_side // 20  # 例如：480 高的帧 → 480/20=24
    win = max(min_size, min(win, max_size))  # 限定范围
    if win % 2 == 0:
        win += 1  # 确保是奇数（SSIM要求）
    return win


def similarity_sampling_kornia_gpu(
    video_path,
    device=torch.device("cuda:0")
):
    from decord import bridge
    bridge.set_bridge("torch")

    # 初始化视频读取
    vr = VideoReader(video_path, ctx=gpu(0))
    native_fps = vr.get_avg_fps()
    total_frames = len(vr)

    # 帧间步长
    step = int(round(native_fps / fps_sampling_rate)) if fps_sampling_rate < native_fps else 1
    sampled_indices = list(range(0, total_frames, step))
    sampled_indices = sampled_indices[:-1]
    if len(sampled_indices) > 60:
        sampled_indices = sampled_indices[:-2]
    if len(sampled_indices) > 300:
        sampled_indices = sampled_indices[:-2]

    # GPU 预处理 transform
    gpu_transform = __build_gpu_transform(device)

    all_frames = []
    all_timestamps = []
    prev_frame_tensor = None

    frame_tensor = vr[0]
    h, w = frame_tensor.shape[0], frame_tensor.shape[1]
    window_size = __get_adaptive_window_size(h, w)
    # print(f"[Auto SSIM] 分辨率: {w}x{h}, 使用 window_size={window_size}")
    with tqdm(total=len(sampled_indices), desc="Processing", unit="frame") as pbar:
        for frame_idx in sampled_indices:
            try:
                # 获取帧: [H, W, 3], uint8, CUDA
                frame_tensor = vr[frame_idx]  # torch.Tensor

                # 转为 [1, 3, H, W] float32
                frame_for_ssim = frame_tensor.permute(2, 0, 1).unsqueeze(0).float() / 255.0  # [1,3,H,W]

                if prev_frame_tensor is not None:
                    ssim_val = kornia.metrics.ssim(
                        prev_frame_tensor, frame_for_ssim,
                        window_size=window_size
                    ).mean()

                    if ssim_val.item() > ssim_threshold:
                        pbar.update(1)
                        continue

                # 应用 transform → 预处理为 [3, 224, 224]
                processed = gpu_transform(frame_for_ssim).squeeze(0)
                all_frames.append(processed)
                all_timestamps.append(round(frame_idx / native_fps,3))

                prev_frame_tensor = frame_for_ssim
                pbar.update(1)

            except Exception as e:
                print(f"[跳过] 处理帧 {frame_idx} 出错: {e}")
                pbar.update(1)
                continue

    if all_frames:
        final_tensor = torch.stack(all_frames, dim=0)  # [N,3,224,224]
        print(f"✅ 保留帧数: {final_tensor.shape[0]}, 尺寸: {final_tensor.shape}")
    else:
        final_tensor = None
        print("❌ 未保留任何帧")

    return all_timestamps, final_tensor



def filter_similar_frames_kornia_gpu(
    frames: list,                   # List[np.ndarray]，每个是 [H, W, 3]，uint8
    timestamps: list,              # List[float]，时间戳（秒）
    keep_frames: list = [],        # 强制保留的时间戳
    ssim_threshold: float = 0.9,
    window_size: int = None,
    device: torch.device = torch.device("cuda:0")
):
    assert len(frames) == len(timestamps)

    filtered_frames = {}
    prev_tensor = None

    if window_size is None and len(frames) > 0:
        h, w = frames[0].shape[:2]
        window_size = __get_adaptive_window_size(h, w)
        # print(f"[Auto SSIM] 分辨率: {w}x{h} → window_size={window_size}")

    for idx, frame_np in enumerate(frames):
        timestamp = timestamps[idx]

        # 转为 [1, 3, H, W] float tensor
        frame_tensor = torch.from_numpy(frame_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        frame_tensor = frame_tensor.to(device)

        # 强制保留帧
        if timestamp in keep_frames:
            image = Image.fromarray(frame_np)
            filtered_frames[timestamp] = image
            prev_tensor = frame_tensor
            continue

        # Kornia SSIM 判断
        if prev_tensor is not None:
            ssim_val = kornia.metrics.ssim(prev_tensor, frame_tensor, window_size=window_size).mean()
            if ssim_val.item() > ssim_threshold:
                continue  # 跳过相似帧

        # 保留原图
        image = Image.fromarray(frame_np)
        filtered_frames[timestamp] = image
        prev_tensor = frame_tensor

    return filtered_frames

if __name__ == "__main__":
    # similarity_sampling(
    #     "../../testVideo/HarryPotterM1.mkv")

    similarity_sampling_kornia_gpu("../../testVideo/HarryPotterM1.mkv")