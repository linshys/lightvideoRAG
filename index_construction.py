import os
import logging
import time
import warnings
import multiprocessing

warnings.filterwarnings("ignore")
logging.getLogger("httpx").setLevel(logging.WARNING)

from lightvideorag import LightVideoRAG

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

    start = time.time()

    # Please enter your video file path in this list; there is no limit on the length.
    # Here is an example; you can use your own videos instead.
    video_paths = [
        'testVideo/Upscaling Face-Off_ PS5 Pro PSSR vs PC DLSS FSR 3.1 in Ratchet and Clank Rift Apart.mp4',
    ]
    videorag = LightVideoRAG(working_dir=f"./lightvideorag-workdir")
    videorag.insert_video(video_path_list=video_paths)

    print(f"total {time.time()-start}s for index construction.")