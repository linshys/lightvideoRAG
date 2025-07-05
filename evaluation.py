import torch
from lightvideorag import LightVideoRAG,QueryParam
import json
from datasets import load_dataset
from collections import defaultdict
from typing import List
import os
import time
import shutil
from tqdm import tqdm
import logging
import warnings
import subprocess
from pathlib import Path


warnings.filterwarnings("ignore")
logging.getLogger("httpx").setLevel(logging.WARNING)

class RAGManager:
    """RAG 索引管理类"""
    def __init__(self):
        self.working_dir = "./lightvideorag-workdir"  # 索引存放目录
        self.videorag = LightVideoRAG(working_dir=self.working_dir)

    def build_index(self, video_path: str):
        self.videorag = LightVideoRAG(working_dir=self.working_dir)
        if isinstance(video_path, str):
            video_path = [video_path]

        self.videorag.insert_video(video_path_list=video_path)
        print(f"✅ 索引已构建: {video_path}")

    def build_index_with_extra(self, video_path: str, extra: dict):
        self.videorag = LightVideoRAG(working_dir=self.working_dir)
        if isinstance(video_path, str):
            video_path = [video_path]

        self.videorag.insert_video(video_path_list=video_path, extra = extra)
        print(f"✅ 索引已构建: {video_path}")

    def rag_inference(self, question: str, options: List[str]) -> str:
        print(f"🔍 运行 RAG 推理: {question}")
        query = f"{question} \nOptions: {', '.join(options)}"
        param = QueryParam(mode="videorag")

        return self.videorag.query(query, param)

    def rag_inference_with_extra(self, question: str, options: List[str], extra:dict) -> str:
        print(f"🔍 运行 RAG 推理: {question}")
        query = f"{question} \nOptions: {'. '.join(options)}"
        param = QueryParam(mode="videorag")
        param.extra_for_rag = extra
        return self.videorag.query(query, param)

    def delete_index(self):
        if os.path.exists(self.working_dir):
            shutil.rmtree(self.working_dir)  # 递归删除目录
            print("✅ 索引文件已删除")
        else:
            print("⚠️ 没有找到索引文件，无需删除")
        self.videorag.unload_model()
        del self.videorag
        self.videorag = None


def evaluate_video_mme(output_file="evaluation_results.json"):
    dataset = load_dataset("lmms-lab/Video-MME")
    video_folder = "video_datasets/video-mme/videos"
    sub_folder = "video_datasets/video-mme/subtitle"

    video_data = defaultdict(lambda: {"duration": None, "domain": None, "sub_category": None, "questions": []})

    # **1. 遍历数据集，按 `video_id` 归类问题**
    for item in dataset["test"]:
        video_id = item["video_id"]

        # 记录视频的元数据（仅在首次出现时赋值）
        if video_data[video_id]["duration"] is None:
            video_data[video_id]["duration"] = item["duration"]
            video_data[video_id]["domain"] = item["domain"]
            video_data[video_id]["sub_category"] = item["sub_category"]
            video_data[video_id]["video_name"] = item["videoID"]

        # **归类问题**
        video_data[video_id]["questions"].append({
            "question_id": item["question_id"],
            "task_type": item["task_type"],
            "question": item["question"],
            "options": item["options"],
            "answer": item["answer"]  # ground truth
        })

    rag_manager = RAGManager()
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_file, "a", encoding="utf-8") as f:
        for idx, (video_id, data) in enumerate(tqdm(video_data.items(), desc="Processing Videos", unit="video")):
            if idx < 0:
                continue  # skip
            video_path = f"{video_folder}/{data['video_name']}.mp4"
            str_path = f"{sub_folder}/{data['video_name']}.srt"

            extra = {
                "task" : "mme",
                "subtitle_path": str_path,
                "video_path": video_path,
            }
            rag_manager.build_index(video_path)

            video_result = {
                "video_id": video_id,
                "duration": data["duration"],
                "domain": data["domain"],
                "sub_category": data["sub_category"],
                "questions": []
            }

            for question in data["questions"]:
                question_id = question["question_id"]
                task_type = question["task_type"]
                question_text = question["question"]
                options = question["options"]

                # model_response = rag_manager.rag_inference_with_extra(question_text, options, extra)
                model_response = rag_manager.rag_inference(question_text, options)

                video_result["questions"].append({
                    "question_id": question_id,
                    "task_type": task_type,
                    "question": question_text,
                    "options": options,
                    "answer": question["answer"],
                    "response": model_response
                })
                torch.cuda.empty_cache()

            f.write(json.dumps(video_result, ensure_ascii=False) + "\n")

            rag_manager.delete_index()

    print(f"Evaluation completed. Results saved to {output_file}")

def evaluete_longvideobench(output_file="evaluation_results.json"):
    data_folder = "video_datasets/longvideobench/LongVideoBench"
    with open(os.path.join(data_folder, "lvb_val.json")) as f:
        dataset = json.load(f)

    video_data = defaultdict(lambda: {
        "duration": None,
        "duration_group": None,
        "subtitle_path": None,
        "video_path": None,
        "starting_timestamp_for_subtitles":None,
        "questions": []
    })
    index_to_abcd = ['A', 'B', 'C', 'D', 'E']

    for item in dataset:
        video_id = item["video_id"]

        if video_data[video_id]["duration"] is None:
            video_data[video_id]["duration"] = item["duration"]
            video_data[video_id]["duration_group"] = item["duration_group"]
            video_data[video_id]["subtitle_path"] = item["subtitle_path"]
            video_data[video_id]["video_path"] = item["video_path"]
            video_data[video_id]["starting_timestamp_for_subtitles"] = item["starting_timestamp_for_subtitles"]


        formatted_candidates = [
            f"{index_to_abcd[i]}. {str(cand)}" for i, cand in enumerate(item["candidates"])
        ]

        correct_choice_letter = index_to_abcd[item["correct_choice"]]

        video_data[video_id]["questions"].append({
            "question_id": item["id"],
            "position": item["position"],
            "question": item["question"],
            "question_wo_referring_query": item["question_wo_referring_query"],
            "question_category": item["question_category"],
            "topic_category": item["topic_category"],
            "level": item["level"],
            "options": formatted_candidates,
            "answer": correct_choice_letter
        })

    rag_manager = RAGManager()
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_file, "a", encoding="utf-8") as f:
        for idx, (video_id, data) in enumerate(tqdm(video_data.items(), desc="Processing Videos", unit="video")):
            if idx < 0:
                continue  # skip
            video_path = os.path.join(data_folder, "videos", data["video_path"])
            subtitle_path = os.path.join(data_folder, "subtitles", data["subtitle_path"])

            extra = {
                "task": "longvideobench",
                "duration": data["duration"],
                "subtitle_path": subtitle_path,
                "video_path": video_path,
                "starting_timestamp_for_subtitles": data['starting_timestamp_for_subtitles']
            }
            rag_manager.build_index_with_extra(video_path,extra)

            video_result = {
                "video_id": video_id,
                "duration": data["duration"],
                "duration_group": data["duration_group"],
                "questions": []
            }

            for question in data["questions"]:
                question_id = question["question_id"]
                question_text = question["question"]
                options = question["options"]

                model_response = rag_manager.rag_inference_with_extra(question_text, options,extra)

                video_result["questions"].append({
                    "question_id": question_id,
                    "question_category": question["question_category"],
                    "topic_category": question["topic_category"],
                    "level": question['level'],
                    "question": question_text,
                    "options": options,
                    "answer": question["answer"],
                    "response": model_response
                })
                torch.cuda.empty_cache()

            f.write(json.dumps(video_result, ensure_ascii=False) + "\n")

            rag_manager.delete_index()

    print(f"Evaluation completed. Results saved to {output_file}")

# ========== 运行评测 ==========
if __name__ == "__main__":
    dataset_name = "lmms-lab/Video-MME"
    # dataset_name = "LongVideoBench"

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    task_name = "rag_3:2:2_uniform10f_mme_window30s_unlimitedlen"

    output_file = f"results/{dataset_name}/{timestamp}--{task_name}.json"

    evaluate_video_mme(output_file= output_file)
    eval_script_path = Path("resolve_and_eval_mme.py")

    # evaluete_longvideobench(output_file= output_file)
    # eval_script_path = Path("resolve_and_eval_long.py")

    subprocess.run([
        "python", str(eval_script_path),
        "--results_file", str(output_file),
        "--return_categories_accuracy",
        "--return_sub_categories_accuracy",
        "--return_task_types_accuracy"
    ])
