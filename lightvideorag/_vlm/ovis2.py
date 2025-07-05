import torch
from transformers import AutoModelForCausalLM
from .prompt import PROMPTS
import datetime

# load model
model = None
text_tokenizer = None
visual_tokenizer = None


def __load_model():
    global model, text_tokenizer, visual_tokenizer
    model = AutoModelForCausalLM.from_pretrained("AIDC-AI/Ovis2-8B",
                                                 torch_dtype=torch.bfloat16,
                                                 multimodal_max_length=32768,
                                                 trust_remote_code=True).cuda()

    model.eval()
    text_tokenizer = model.get_text_tokenizer()
    visual_tokenizer = model.get_visual_tokenizer()

def ovis2_unload():
    __unload_model()

def __unload_model():
    global model, text_tokenizer, visual_tokenizer
    if model is not None:
        del model, text_tokenizer, visual_tokenizer  # 删除变量
        model = None
        text_tokenizer = None
        visual_tokenizer = None
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def ovis2_inference(
        query: str,
        frames: [],
        transcripts: []
) -> str:
    global model, text_tokenizer, visual_tokenizer
    if model is None:
        __load_model()

    max_partition = 1
    pre_query = "Video Clips retrieved from the whole video:\n"
    image_list = []
    for idx in range(len(frames)):
        pre_query += "\n".join(
            f"{seconds_to_hms(time)}: <image>" for time in frames[idx]
        )
        image_list.extend(frames[idx].values())
        if transcripts[idx].strip():
            pre_query += '\nCorresponding Transcript: ' + transcripts[idx] + '\n'
        else:
            pre_query += '\n'

        pre_query += 'Next chunk: \n'

    pre_query += "\nQuery:\n" + query

    # instructions_prompt = PROMPTS['video_long_inference']  # please use this prompt template if you want to do QA evaluation
    instructions_prompt = PROMPTS['common_inference']
    conversion = [
        {
            "from": "system",
            "value": instructions_prompt
        },
        {
            "from": "human",
            "value": pre_query
        }
    ]

    # format conversation
    prompt, input_ids, pixel_values = model.preprocess_inputs(conversion, image_list, max_partition=max_partition)
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


def seconds_to_hms(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))


def ovis2_refine_retrieval_query(query: str):
    global model, text_tokenizer, visual_tokenizer
    if model is None:
        __load_model()

    try:
        refined_prompt = {
            'visual_retrieval': __query_rewrite_for_multi_retrieval(query, "query_rewrite_for_visual_retrieval"),
            'transcript_retrieval': __query_rewrite_for_multi_retrieval(query, "query_rewrite_for_transcript_retrieval")
        }
    finally:
        __unload_model()

    return refined_prompt


def __query_rewrite_for_multi_retrieval(query: str, prompt_template: str) -> str:
    images = []
    max_partition = None

    query_rewrite_prompt = PROMPTS[prompt_template]
    conversion = [
        {
            "from": "system",
            "value": query_rewrite_prompt
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


if __name__ == "__main__":
    print(ovis2_refine_retrieval_query("What did Harry first see in the Mirror of Erised."))
