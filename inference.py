import os
import logging
import warnings
import multiprocessing
import time

warnings.filterwarnings("ignore")
logging.getLogger("httpx").setLevel(logging.WARNING)

from lightvideorag import QueryParam
from lightvideorag import LightVideoRAG


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

    start = time.time()

    query = 'Does FSR provide better visual detail?'
    param = QueryParam(mode="videorag")

    videorag = LightVideoRAG(working_dir=f"./lightvideorag-workdir")
    response = videorag.query(query=query, param=param)
    print(response)
    print(f"{time.time()-start}s to inference")