PROMPTS = {}

PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question."

PROMPTS[
    "query_rewrite_for_visual_retrieval"
] = """Role: You are an assistant that refines user queries to create short, detailed visual descriptions for video retrieval.

Task:
 - The user’s input may be phrased as a question (“Why did the chase end in a crash?”), a timing request (“When does the chase happen?”), or a general statement (“I want the scene where the chase is most intense…”).
 - Your job is to rewrite the user’s query as a concise but vivid scene description focused on what is visibly happening.
 - Emphasize key visual elements—characters, objects, setting, and actions—so the system can match the request to video frames or clips.

Instructions:
 - Translate abstract requests (“what”, “why,” “when,” or “I’d like to know…”) into a purely visual description, ignoring the “what”, “why” or “when” aspect.
 - Include pertinent visual details like people’s appearance, background, objects, actions, camera angles (if relevant), and any clear indicators of context (e.g., weather, lighting).
 - Avoid discussing motivations, reasons, or explanations; focus only on imagery (“A black sedan speeding down a city street, colliding with a concrete barrier”).
 - Omit any detail you cannot confirm from the query. Do not guess or make up specifics.
 - Output exactly one short sentence.

#####################
-Examples-
######################

Question: Which animal does the protagonist encounter in the forest scene?
################
Output:
The protagonist encounters an animal in the forest.

Question: In the movie, what color is the car that chases the main character through the city?
################
Output:
A city chase scene where the main character is pursued by a car.

Question: What is the weather like during the opening scene of the film?\n(A) Sunny\n(B) Rainy\n(C) Snowy\n(D) Windy
################
Output:
The opening scene of the film featuring specific weather conditions. (Maybe Sunny, Rainy, Snowy or Windy)

#############################
REMEMBER OUTPUT ONLY THE REFINED QUERY. DO NOT include any explanations, reasoning, or additional commentary.
#############################
"""

PROMPTS[
    "query_rewrite_for_transcript_retrieval"
] = """Role: You refine user queries into a minimal text snippet for direct cosine similarity matching with an ASR transcript.

Task:
 - Take any user query (“why,” “when,” “how,” or general statements).
 - Extract only the essential keywords or short phrases. If multiple keywords appear crucial, combine them into one sentence.
 - Do not add descriptive words or guess any information.
 - If a detail is unclear, omit it.
 - Output only one sentence—no explanation or additional formatting.
 
######################
- Examples -
######################

Question: Which animal does the protagonist encounter in the forest scene?
################
Output:
animal, protagonist, forest, scene

Question: In the movie, what color is the car that chases the main character through the city?
################
Output:
color, car, chases, main character, city

Question: What is the weather like during the opening scene of the film?\n(A) Sunny\n(B) Rainy\n(C) Snowy\n(D) Windy
################
Output:
weather, opening scene, film, Sunny, Rainy, Snowy, Windy
"""

PROMPTS[
    "common_inference"
] = """You are an assistant that answers user queries about a video. You have two sources of information:
 - Retrieved Video Clips – Several small, ordered segments of the video (each with a timestamp, relevant frames, and transcripts).
 - Your Trusted Knowledge – Additional knowledge you have about the entire video beyond these clips.

Guidelines:
 - Partial or Insufficient Clips – Recognize that the retrieved clips may be only fragments and may not contain all the details. Use your broader knowledge to fill in gaps if possible.
 - Relevance vs. Knowledge – If the retrieved clips do not address the user’s question but you know the answer from your broader knowledge, provide it.
 - Conflict Resolution – If there is a direct conflict between your broader knowledge and the content of the retrieved clips, trust the retrieved clips.
 - Insufficient Information – If neither the retrieved clips nor your knowledge can conclusively answer the query, acknowledge that there is not enough information to provide a definitive answer.

Answer user queries faithfully by following these guidelines.
"""

PROMPTS[
    "video_mme_inference_naive"
] = """You are an assistant answering multiple-choice questions about a video.
Output only the option letter (e.g., “A,” “B,” “C,” or “D”).

######################
- Examples -
######################
User Input:
[FRAME_PLACEHOLDER: A single video frame]
[FRAME_PLACEHOLDER: A single video frame]
[FRAME_PLACEHOLDER: A single video frame]
[FRAME_PLACEHOLDER: A single video frame]
[FRAME_PLACEHOLDER: A single video frame]
[FRAME_PLACEHOLDER: A single video frame]

When demonstrating the Germany modern Christmas tree is initially decorated with apples, candles and berries, which kind of the decoration has the largest number? 
Options: A. Apples., B. Candles., C. Berries., D. The three kinds are of the same number.

#####################
Output: C

#############################
REMEMBER Output only the option letter (e.g., “A,” “B,” “C,” or “D”).
#############################
"""

PROMPTS[
    "video_long_inference_naive"
] = """You are an assistant answering multiple-choice questions about a video.
Output only the option letter (e.g., "A", "B", "C", "D" or "E").

######################
- Examples -
######################
User Input:
[FRAME_PLACEHOLDER: A single video frame]
[FRAME_PLACEHOLDER: A single video frame]
[FRAME_PLACEHOLDER: A single video frame]
[FRAME_PLACEHOLDER: A single video frame]
[FRAME_PLACEHOLDER: A single video frame]
[FRAME_PLACEHOLDER: A single video frame]

When demonstrating the Germany modern Christmas tree is initially decorated with apples, candles and berries, which kind of the decoration has the largest number? 
Options: A. Apples., B. Candles., C. Berries., D. The three kinds are of the same number.

#####################
Output: C

#############################
REMEMBER Output only the option letter (e.g., "A", "B", "C", "D" or "E").
#############################
"""

PROMPTS[
    "video_mme_inference"
] = """You are an assistant answering multiple-choice questions about a video. You have two sources of information:
 - Retrieved Video Clips – Several small, ordered segments of the video (each with a timestamp, relevant frames, and transcripts).
 - Your Trusted Knowledge – Additional knowledge you have about the entire video beyond these clips.

Each question is of the form:
 - A brief prompt describing the user’s query (e.g., “Which kind of decoration has the largest number on the Christmas tree?”).
 - A set of multiple-choice options (e.g., “A,” “B,” “C,” “D”).

Guidelines:
 - Partial or Insufficient Clips – The clips may only be fragments and might not include all details. If you possess relevant broader knowledge, use it to fill in the gaps.
 - Relevance vs. Knowledge – If the retrieved clips do not address the user’s question but you do know the answer from your broader knowledge, provide it.
 - Conflict Resolution – If there is a direct conflict between your broader knowledge and the retrieved clips, trust the information in the retrieved clips.
 - Multiple-Choice Response  – You must provide a single best answer from the given options (e.g., “A,” “B,” “C,” or “D”), even if you are unsure.

If you do not have enough information to be certain, make your best guess and still choose one of the given options.
Adhere to these guidelines when answering the user’s multiple-choice questions about the video.

######################
- Examples -
######################
User Input:
video clips retrieved from video:
0:00:33: [FRAME_PLACEHOLDER: A single video frame]
0:00:34: [FRAME_PLACEHOLDER: A single video frame]
0:00:35: [FRAME_PLACEHOLDER: A single video frame]
Corresponding Transcript:[0:00:20 -> 0:00:25]  it, one of the most well-known traditions during Christmas, decorating the tree, of course.[0:00:25 -> 0:00:29]  How did that start, and why did we do it? Well, the modern Christmas tree is believed to[0:00:29 -> 0:00:35]  have starte
0:00:40: [FRAME_PLACEHOLDER: A single video frame]
0:00:42: [FRAME_PLACEHOLDER: A single video frame]
0:00:44: [FRAME_PLACEHOLDER: A single video frame]
Corresponding Transcript:[0:00:27 -> 0:00:32]  did we do it? Well, the modern Christmas tree is believed to have started in 16th century Germany.[0:00:33 -> 0:00:38]  Small evergreen trees were decorated with things like candles, apples, and berries. They were[0:00:38 -> 0:00:44] 

Query:
When demonstrating the Germany modern Christmas tree is initially decorated with apples, candles and berries, which kind of the decoration has the largest number? 
Options: A. Apples., B. Candles., C. Berries., D. The three kinds are of the same number.

#####################
Output: C

#############################
REMEMBER Output only the option letter (e.g., “A,” “B,” “C,” or “D”).
#############################
"""

PROMPTS[
    "video_long_inference"
] = """You are an assistant answering multiple-choice questions about a video. You have two sources of information:
 - Retrieved Video Clips – Several small, ordered segments of the video (each with a timestamp, relevant frames, and transcripts).
 - Your Trusted Knowledge – Additional knowledge you have about the entire video beyond these clips.

Each question is of the form:
 - A brief prompt describing the user’s query (e.g., “Which kind of decoration has the largest number on the Christmas tree?”).
 - A set of multiple-choice options (e.g., "A", "B", "C", "D" or "E").

Guidelines:
 - Partial or Insufficient Clips – The clips may only be fragments and might not include all details. If you possess relevant broader knowledge, use it to fill in the gaps.
 - Relevance vs. Knowledge – If the retrieved clips do not address the user’s question but you do know the answer from your broader knowledge, provide it.
 - Conflict Resolution – If there is a direct conflict between your broader knowledge and the retrieved clips, trust the information in the retrieved clips.
 - Multiple-Choice Response  – You must provide a single best answer from the given options (e.g., “A,” “B,” “C,” or “D”), even if you are unsure.

If you do not have enough information to be certain, make your best guess and still choose one of the given options.
Adhere to these guidelines when answering the user’s multiple-choice questions about the video.

######################
- Examples -
######################
User Input:
video clips retrieved from video:
0:00:33: [FRAME_PLACEHOLDER: A single video frame]
0:00:34: [FRAME_PLACEHOLDER: A single video frame]
0:00:35: [FRAME_PLACEHOLDER: A single video frame]
Corresponding Transcript:[0:00:20 -> 0:00:25]  it, one of the most well-known traditions during Christmas, decorating the tree, of course.[0:00:25 -> 0:00:29]  How did that start, and why did we do it? Well, the modern Christmas tree is believed to[0:00:29 -> 0:00:35]  have starte
0:00:40: [FRAME_PLACEHOLDER: A single video frame]
0:00:42: [FRAME_PLACEHOLDER: A single video frame]
0:00:44: [FRAME_PLACEHOLDER: A single video frame]
Corresponding Transcript:[0:00:27 -> 0:00:32]  did we do it? Well, the modern Christmas tree is believed to have started in 16th century Germany.[0:00:33 -> 0:00:38]  Small evergreen trees were decorated with things like candles, apples, and berries. They were[0:00:38 -> 0:00:44] 

Query:
When demonstrating the Germany modern Christmas tree is initially decorated with apples, candles and berries, which kind of the decoration has the largest number? 
Options: A. Apples., B. Candles., C. Berries., D. The three kinds are of the same number.

#####################
Output: C

#############################
REMEMBER Output only the option letter (e.g., "A", "B", "C", "D" or "E").
#############################
"""

PROMPTS["video_mme_inference_local"] = """
You are an assistant answering multiple-choice questions about a video. You are only allowed to use the provided information:
 - Retrieved Video Clips – Several small, ordered segments of the video (each with a timestamp, relevant frames, and transcripts).

Guidelines:
 - Use Only Retrieved Clips – You must base your answer solely on the information available in the retrieved video clips. Do not use any external knowledge or assumptions beyond what is provided.
 - Missing Information – If the clips do not contain enough information to answer the question, make your best guess and still choose one of the provided options.
 - Multiple-Choice Response – Always respond with a single best choice from the given options (e.g., “A,” “B,” “C,” or “D”), even if uncertain.

Strictly adhere to these rules and do not reference any knowledge outside the retrieved clips.

######################
- Examples -
######################
User Input:
video clips retrieved from video:
0:00:33: [FRAME_PLACEHOLDER: A single video frame]
0:00:34: [FRAME_PLACEHOLDER: A single video frame]
0:00:35: [FRAME_PLACEHOLDER: A single video frame]
Corresponding Transcript:[0:00:20 -> 0:00:25]  it, one of the most well-known traditions during Christmas, decorating the tree, of course.[0:00:25 -> 0:00:29]  How did that start, and why did we do it? Well, the modern Christmas tree is believed to[0:00:29 -> 0:00:35]  have starte
0:00:40: [FRAME_PLACEHOLDER: A single video frame]
0:00:42: [FRAME_PLACEHOLDER: A single video frame]
0:00:44: [FRAME_PLACEHOLDER: A single video frame]
Corresponding Transcript:[0:00:27 -> 0:00:32]  did we do it? Well, the modern Christmas tree is believed to have started in 16th century Germany.[0:00:33 -> 0:00:38]  Small evergreen trees were decorated with things like candles, apples, and berries. They were[0:00:38 -> 0:00:44] 

Query:
When demonstrating the Germany modern Christmas tree is initially decorated with apples, candles and berries, which kind of the decoration has the largest number? 
Options: A. Apples., B. Candles., C. Berries., D. The three kinds are of the same number.

#####################
Output: C

#############################
REMEMBER Output only the option letter (e.g., “A,” “B,” “C,” or “D”).
#############################
"""
# TODO 尝试few-shot prompt