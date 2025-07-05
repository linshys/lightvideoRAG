import os
from collections import defaultdict
from pathlib import Path
import argparse
import json
from typing import List, Dict, Optional, Union
import re

DURATION_GROUPS = [15, 60, 600, 3600]

def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is"
        "The correct option is",
        "Best answer:"
        "Best option:",
        "Answer:",
        "Option:",
        "The correct answer",
        "The correct option",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    if len(s.split()) > 10 and not re.search("[ABCDE]", s):
        return ""
    # 更严格的正则表达式，匹配单独的 A/B/C/D
    matches = re.findall(r'\b[A-E]\b', s)
    if matches is None:
        return ""
    return matches[0]


def eval_your_results(
        your_results_path: str,
        duration_groups: Optional[Union[List[int], str]] = None,
        return_categories_accuracy: Optional[bool] = True,
        return_topic_category_accuracy: Optional[bool] = False,
        return_question_category_accuracy: Optional[bool] = False,
        gt_answer_key: Optional[str] = "answer",
        your_answer_key: Optional[str] = "response"
):
    if isinstance(duration_groups, str):
        duration_groups = [int(x) for x in duration_groups.split(",")]
    elif duration_groups is None:
        duration_groups = DURATION_GROUPS

    with open(your_results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    q_type_dict = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "answered": 0}))
    topic_dict = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "answered": 0}))

    for item in data:
        group = item.get("duration_group")
        if group not in duration_groups:
            continue

        for question in item.get("questions", []):
            q_type = question.get("question_category")
            topic = question.get("topic_category")

            gt = question.get(gt_answer_key)
            pred = extract_characters_regex(question.get(your_answer_key, ""))

            if pred:
                q_type_dict[group][q_type]["answered"] += 1
                q_type_dict[group][q_type]["correct"] += int(pred == gt)

                topic_dict[group][topic]["answered"] += 1
                topic_dict[group][topic]["correct"] += int(pred == gt)

    for group in duration_groups:
        print(f"\n========== Duration Group: {group} ==========")

        if return_question_category_accuracy:
            print("\n-- Question Category Accuracy --")
            for q_type, stats in q_type_dict[group].items():
                a, c = stats["answered"], stats["correct"]
                print(f"{q_type:20s}: {100 * c / a if a else 0:.1f}%")

        if return_topic_category_accuracy:
            print("\n-- Topic Category Accuracy --")
            for topic, stats in topic_dict[group].items():
                a, c = stats["answered"], stats["correct"]
                print(f"{topic:30s}: {100 * c / a if a else 0:.1f}%")

        group_correct = sum(q_type_dict[group][q]["correct"] for q in q_type_dict[group])
        group_answered = sum(q_type_dict[group][q]["answered"] for q in q_type_dict[group])

        print(f"Overall: {100 * group_correct / group_answered if group_answered else 0:.1f}%")

    total_correct = sum(q_type_dict[g][q]["correct"] for g in duration_groups for q in q_type_dict[g])
    total_answered = sum(q_type_dict[g][q]["answered"] for g in duration_groups for q in q_type_dict[g])
    print("\n========== Overall Accuracy ==========")
    print(f"Overall: {100 * total_correct / total_answered if total_answered else 0:.1f}%")

def jsonl_to_json(input_file: Path, output_file):
    data = []
    with open(input_file, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            line = line.strip()
            if line:
                try:
                    obj = json.loads(line)
                    data.append(obj)
                except json.JSONDecodeError as e:
                    print(f"⚠️ 跳过解析失败的行: {e}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f_out:
        json.dump(data, f_out, ensure_ascii=False, indent=2)

import sys
import contextlib

class Tee(object):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", type=str, required=True)
    parser.add_argument("--duration_groups", type=int, nargs="+", default=[15, 60, 600, 3600])
    parser.add_argument("--return_categories_accuracy", action="store_true")
    parser.add_argument("--return_topic_category_accuracy", action="store_true")
    parser.add_argument("--return_question_category_accuracy", action="store_true")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    input_path = script_dir / args.results_file

    output_file = script_dir / f"results/resolved/{input_path.stem}.json"

    jsonl_to_json(input_path, output_file)

    txt_output_file = output_file.with_suffix(".txt")
    with open(txt_output_file, 'w', encoding='utf-8') as f:
        tee = Tee(sys.stdout, f)
        with contextlib.redirect_stdout(tee):
            eval_your_results(
                str(output_file),
                duration_groups=args.duration_groups,
                return_categories_accuracy=args.return_categories_accuracy,
                return_topic_category_accuracy=args.return_topic_category_accuracy,
                return_question_category_accuracy=args.return_question_category_accuracy
            )

