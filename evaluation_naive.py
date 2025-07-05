from logging import exception

import torch
from PIL import Image
from transformers import AutoModelForCausalLM

import json
from datasets import load_dataset
from collections import defaultdict
import os
import time
from tqdm import tqdm
import logging
import warnings

from video_datasets import LongVideoBenchDataset, slice_frames
from lightvideorag._vlm.prompt import PROMPTS

warnings.filterwarnings("ignore")
logging.getLogger("httpx").setLevel(logging.WARNING)

def evaluate_mme_naive(output_file="evaluation_results_mme_naive.json"):
    dataset = load_dataset("lmms-lab/Video-MME")
    video_folder = "video_datasets/video-mme/videos"
    sub_folder = "video_datasets/video-mme/subtitle"

    model, text_tokenizer, visual_tokenizer = ovis2_load()

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

    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_file, "a", encoding="utf-8") as f:
        for idx, (video_id, data) in enumerate(tqdm(video_data.items(), desc="Processing Videos", unit="video")):
            if idx < 0:
                continue  # skip
            video_path = f"{video_folder}/{data['video_name']}.mp4"
            str_path = f"{sub_folder}/{data['video_name']}.srt"

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

                user_query =f"{question_text} \nOptions: {', '.join(options)}"
                # model_response = ovis2_inference_naive(user_query,video_path, model, text_tokenizer, visual_tokenizer)

                # eval with sub
                frames, subtitles =  slice_frames(video_path, str_path, num_frames=32)
                model_response = ovis2_inference_naive_mme_w_sub(user_query, frames, subtitles, model, text_tokenizer, visual_tokenizer)

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


    print(f"Evaluation completed. Results saved to {output_file}")

def ovis2_inference_naive(user_query, video_path, model, text_tokenizer, visual_tokenizer):
    from moviepy.editor import VideoFileClip
    num_frames = 12
    max_partition = 1

    with VideoFileClip(video_path) as clip:
        total_frames = int(clip.fps * clip.duration)
        if total_frames <= num_frames:
            sampled_indices = range(total_frames)
        else:
            stride = total_frames / num_frames
            sampled_indices = [min(total_frames - 1, int((stride * i + stride * (i + 1)) / 2)) for i in range(num_frames)]
        frames = [clip.get_frame(index / clip.fps) for index in sampled_indices]
        frames = [Image.fromarray(frame, mode='RGB') for frame in frames]
    images = frames
    query = '\n'.join(['<image>'] * len(images)) + '\n' + user_query

    instructions_prompt = PROMPTS['video_long_inference_naive']
    conversion = [
        {
            "from": "system",
            "value": instructions_prompt
        },
        {
            "from": "human",
            "value": query
        }
    ]

    # format conversation
    prompt, input_ids, pixel_values = model.preprocess_inputs(conversion, images, max_partition=max_partition)
    attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
    input_ids = input_ids.unsqueeze(0).to(device=model.device)
    attention_mask = attention_mask.unsqueeze(0).to(device=model.device)
    if pixel_values is not None:
        pixel_values = pixel_values.to(dtype=visual_tokenizer.dtype, device=visual_tokenizer.device)
    pixel_values = [pixel_values]

    # generate output
    with torch.inference_mode():
        gen_kwargs = dict(
            max_new_tokens=1024,
            do_sample=False,
            top_p=None,
            top_k=None,
            temperature=None,
            repetition_penalty=None,
            eos_token_id=model.generation_config.eos_token_id,
            pad_token_id=text_tokenizer.pad_token_id,
            use_cache=True
        )
        output_ids = model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, **gen_kwargs)[
            0]
        output = text_tokenizer.decode(output_ids, skip_special_tokens=True)
        return output

def ovis2_load():
    # load model
    model = AutoModelForCausalLM.from_pretrained("AIDC-AI/Ovis2-8B",
                                                 torch_dtype=torch.bfloat16,
                                                 multimodal_max_length=32768,
                                                 trust_remote_code=True).cuda()
    text_tokenizer = model.get_text_tokenizer()
    visual_tokenizer = model.get_visual_tokenizer()

    return model, text_tokenizer, visual_tokenizer

def evaluate_long_naive(output_file="evaluation_results_long_naive.json"):
    model, text_tokenizer, visual_tokenizer = ovis2_load()

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

    for idx, item in enumerate(dataset):
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
            "idx": idx,
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

    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_file, "a", encoding="utf-8") as f:
        dataset = LongVideoBenchDataset(data_folder, "lvb_val.json", max_num_frames=32)
        for idx, (video_id, data) in enumerate(tqdm(video_data.items(), desc="Processing Videos", unit="video")):
            if idx < 0:
                continue  # skip

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

                # if question_id != dataset[question["idx"]]['id']:
                #     raise exception("数据不匹配")
                # inputs =  dataset[question["idx"]]['inputs']
                # model_response = ovis2_inference_naive_long(inputs, model, text_tokenizer, visual_tokenizer)

                user_query = f"{question_text} \nOptions: {', '.join(options)}"
                video_path = f"{data_folder}/videos/{data['video_path']}"
                model_response = ovis2_inference_naive(user_query, video_path, model, text_tokenizer, visual_tokenizer)

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

    print(f"Evaluation completed. Results saved to {output_file}")


def ovis2_inference_naive_long(inputs: list, model, text_tokenizer, visual_tokenizer):
    max_partition = 1
    pre_query = "Video Clips retrieved from the whole video:\n"

    image_list = []

    # 遍历处理
    for item in inputs:
        if isinstance(item, Image.Image):  # 如果是图像
            pre_query += "<image>"
            image_list.append(item)
        elif isinstance(item, str):  # 如果是字符串
            pre_query +=  item+ "\n"
        else:
            raise TypeError(f"不支持的类型: {type(item)}")

    # format conversation
    prompt, input_ids, pixel_values = model.preprocess_inputs(pre_query, image_list, max_partition=max_partition)
    attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
    input_ids = input_ids.unsqueeze(0).to(device=model.device)
    attention_mask = attention_mask.unsqueeze(0).to(device=model.device)
    if pixel_values is not None:
        pixel_values = pixel_values.to(dtype=visual_tokenizer.dtype, device=visual_tokenizer.device)
    pixel_values = [pixel_values]

    # generate output
    with torch.inference_mode():
        gen_kwargs = dict(
            max_new_tokens=1024,
            do_sample=False,
            top_p=None,
            top_k=None,
            temperature=None,
            repetition_penalty=None,
            eos_token_id=model.generation_config.eos_token_id,
            pad_token_id=text_tokenizer.pad_token_id,
            use_cache=True
        )
        output_ids = model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, **gen_kwargs)[
            0]
        output = text_tokenizer.decode(output_ids, skip_special_tokens=True)
        return output

def ovis2_inference_naive_mme_w_sub(user_query:str, frames, subtitles, model, text_tokenizer, visual_tokenizer):
    max_partition = 1
    pre_query = "Video Clips retrieved from the whole video:"
    pre_query += '\n'.join(['<image>'] * len(frames))
    pre_query += '\n' + "This video's subtitles are listed below:\n" + subtitles
    pre_query += '\n' + "Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.\n" + user_query
    pre_query += '\n' + "The best answer is:"

    # format conversation
    prompt, input_ids, pixel_values = model.preprocess_inputs(pre_query, frames, max_partition=max_partition)
    attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
    input_ids = input_ids.unsqueeze(0).to(device=model.device)
    attention_mask = attention_mask.unsqueeze(0).to(device=model.device)
    if pixel_values is not None:
        pixel_values = pixel_values.to(dtype=visual_tokenizer.dtype, device=visual_tokenizer.device)
    pixel_values = [pixel_values]

    # generate output
    with torch.inference_mode():
        gen_kwargs = dict(
            max_new_tokens=1024,
            do_sample=False,
            top_p=None,
            top_k=None,
            temperature=None,
            repetition_penalty=None,
            eos_token_id=model.generation_config.eos_token_id,
            pad_token_id=text_tokenizer.pad_token_id,
            use_cache=True
        )
        output_ids = model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, **gen_kwargs)[
            0]
        output = text_tokenizer.decode(output_ids, skip_special_tokens=True)
        return output
if __name__ == "__main__":
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    evaluate_mme_naive(output_file=f"results/lmms-lab/Video-MME/naive/{timestamp}_naive_mme.json")
    # evaluate_long_naive(output_file=f"results/LongVideoBench/naive/{timestamp}_naive_long.json")
