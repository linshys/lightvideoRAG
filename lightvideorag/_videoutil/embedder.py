import torch
from tqdm import tqdm
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

from ImageBind.imagebind import data

# 定义 batch_size
FRAME_BATCH_SIZE = 512
TEXT_BATCH_SIZE = 512

def embedder_image_bind_frames(frames_tensor):
    if isinstance(frames_tensor, list):
        frames_tensor = torch.stack(frames_tensor)  # 转换成 Tensor
    elif not isinstance(frames_tensor, torch.Tensor):
        raise TypeError("final_tensor 必须是 torch.Tensor 或 可转换的 list")

    # 获取总数据量
    total_samples = frames_tensor.shape[0]
    num_batches = (total_samples + FRAME_BATCH_SIZE - 1) // FRAME_BATCH_SIZE  # 计算总批次数量

    # 加载 ImageBind 模型
    embedder = imagebind_model.imagebind_huge(pretrained=True).cuda()
    embedder.eval()

    # 存储所有的 batch 结果
    all_embeddings = []

    try:
        # 按 batch 进行推理，并用 tqdm 记录进度
        with torch.no_grad():
            for start_idx in tqdm(range(0, total_samples, FRAME_BATCH_SIZE), total=num_batches, desc="Processing Batches"):
                end_idx = min(start_idx + FRAME_BATCH_SIZE, total_samples)
                batch_tensor = frames_tensor[start_idx:end_idx].cuda()  # 取 batch 并移动到 GPU

                # 构造输入字典
                inputs = {
                    ModalityType.VISION: batch_tensor
                }

                # 获取嵌入
                batch_embeddings = embedder(inputs)

                # 存储到列表（先转移到 CPU 避免显存溢出）
                all_embeddings.append(batch_embeddings[ModalityType.VISION].cpu())

        # 合并所有 batch 结果
        all_embeddings = torch.cat(all_embeddings, dim=0)

        return all_embeddings.numpy()
    finally:
        del embedder  # 删除模型对象
        torch.cuda.empty_cache()  # 释放 GPU 显存


def embedder_image_bind_transcript(transcripts):
    if not transcripts:
        print("[Warning] No transcripts provided. Skipping embedding.")
        return []  # 或者 return None，取决于后续逻辑
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_samples = len(transcripts)
    num_batches = (total_samples + TEXT_BATCH_SIZE - 1) // TEXT_BATCH_SIZE  # 计算总批次数量

    embedder = imagebind_model.imagebind_huge(pretrained=True).cuda()
    embedder.eval()

    all_embeddings = []

    try:
        with torch.no_grad():
            for start_idx in tqdm(range(0, total_samples, TEXT_BATCH_SIZE), total=num_batches, desc="Processing Transcript Batches"):
                end_idx = min(start_idx + TEXT_BATCH_SIZE, total_samples)
                batch_text = [item["text"] for item in transcripts[start_idx:end_idx]]

                inputs = {
                    ModalityType.TEXT: data.load_and_transform_text(batch_text, device)
                }

                batch_embeddings = embedder(inputs)
                all_embeddings.append(batch_embeddings[ModalityType.TEXT].cpu())

        # 合并所有 batch 结果
        if not all_embeddings:
            print("[Warning] No embeddings to concatenate.")
            return []
        all_embeddings = torch.cat(all_embeddings, dim=0)

        return all_embeddings.numpy()
    finally:
        del embedder
        torch.cuda.empty_cache()
