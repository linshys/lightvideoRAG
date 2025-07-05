import asyncio
import os
import torch
from dataclasses import dataclass
import numpy as np
from nano_vectordb import NanoVectorDB
from tqdm import tqdm
from imagebind.models import imagebind_model

from .._utils import logger
from ..base import BaseVectorStorage
from .._videoutil import encode_video_segments, encode_string_query


@dataclass
class NanoVectorDBStorage(BaseVectorStorage):
    cosine_better_than_threshold: float = 0.2
    
    def __post_init__(self):

        self._client_file_name = os.path.join(
            self.global_config["working_dir"], f"vdb_{self.namespace}.json"
        )
        self._max_batch_size = self.global_config["embedding_batch_num"]
        self._client = NanoVectorDB(
            self.embedding_func.embedding_dim, storage_file=self._client_file_name
        )
        self.cosine_better_than_threshold = self.global_config.get(
            "query_better_than_threshold", self.cosine_better_than_threshold
        )

    async def upsert(self, data: dict[str, dict]):
        logger.info(f"Inserting {len(data)} vectors to {self.namespace}")
        if not len(data):
            logger.warning("You insert an empty data to vector DB")
            return []
        list_data = [
            {
                "__id__": k,
                **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
            }
            for k, v in data.items()
        ]
        contents = [v["content"] for v in data.values()]
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]
        embeddings_list = await asyncio.gather(
            *[self.embedding_func(batch) for batch in batches]
        )
        embeddings = np.concatenate(embeddings_list)
        for i, d in enumerate(list_data):
            d["__vector__"] = embeddings[i]
        results = self._client.upsert(datas=list_data)
        return results

    async def query(self, query: str, top_k=5):
        embedding = await self.embedding_func([query])
        embedding = embedding[0]
        results = self._client.query(
            query=embedding,
            top_k=top_k,
            better_than_threshold=self.cosine_better_than_threshold,
        )
        results = [
            {**dp, "id": dp["__id__"], "distance": dp["__metrics__"]} for dp in results
        ]
        return results

    async def index_done_callback(self):
        self._client.save()


@dataclass
class NanoVectorDBVideoSegmentStorage(BaseVectorStorage):
    embedding_func = None
    segment_retrieval_top_k: float = 2
    
    def __post_init__(self):
        
        self._client_file_name = os.path.join(
            self.global_config["working_dir"], f"vdb_{self.namespace}.json"
        )
        self._max_batch_size = self.global_config["video_embedding_batch_num"]
        self._client = NanoVectorDB(
            self.global_config["video_embedding_dim"], storage_file=self._client_file_name
        )
        self.top_k = self.global_config.get(
            "segment_retrieval_top_k", self.segment_retrieval_top_k
        )
    
    # async def upsert(self, video_name, segment_index2name, video_output_format):
    #     embedder = imagebind_model.imagebind_huge(pretrained=True).cuda()
    #     embedder.eval()
    #
    #     logger.info(f"Inserting {len(segment_index2name)} segments to {self.namespace}")
    #     if not len(segment_index2name):
    #         logger.warning("You insert an empty data to vector DB")
    #         return []
    #     list_data, video_paths = [], []
    #     cache_path = os.path.join(self.global_config["working_dir"], '_cache', video_name)
    #     index_list = list(segment_index2name.keys())
    #     for index in index_list:
    #         list_data.append({
    #             "__id__": f"{video_name}_{index}",
    #             "__video_name__": video_name,
    #             "__index__": index,
    #         })
    #         segment_name = segment_index2name[index]
    #         video_file = os.path.join(cache_path, f"{segment_name}.{video_output_format}")
    #         video_paths.append(video_file)
    #     batches = [
    #         video_paths[i: i + self._max_batch_size]
    #         for i in range(0, len(video_paths), self._max_batch_size)
    #     ]
    #     embeddings = []
    #     for _batch in tqdm(batches, desc=f"Encoding Video Segments {video_name}"):
    #         batch_embeddings = encode_video_segments(_batch, embedder)
    #         embeddings.append(batch_embeddings)
    #     embeddings = torch.concat(embeddings, dim=0)
    #     embeddings = embeddings.numpy()
    #     for i, d in enumerate(list_data):
    #         d["__vector__"] = embeddings[i]
    #     results = self._client.upsert(datas=list_data)
    #     return results

    async def upsert_frames(self, video_name, frame_embeddings, timestamp):
        logger.info(f"Inserting {len(frame_embeddings)} frames to {self.namespace}")
        if not len(frame_embeddings):
            logger.warning("You insert an empty data to vector DB")
            return []

        list_data = []

        start_index = self._client.__len__()
        for idx, embedding in enumerate(frame_embeddings):
            list_data.append({
                "__id__": f"{video_name}_v_{start_index + idx}",
                "__video_name__": video_name,
                "__index__": f"v_{start_index + idx}",
                "__timestamp__": timestamp[idx],
                "__vector__": embedding
            })

        results = self._client.upsert(datas=list_data)
        return results


    async def upsert_transcripts(self, video_name, transcripts, transcript_embeddings):
        logger.info(f"Inserting {len(transcripts)} transcripts to {self.namespace}")
        if not len(transcripts):
            logger.warning("You insert an empty data to vector DB")
            return []

        list_data = []

        start_index = self._client.__len__()
        for idx, transcript in enumerate(transcripts):
            list_data.append({
                "__id__": f"{video_name}_t_{start_index + idx}",
                "__video_name__": video_name,
                "__index__": f"t_{start_index + idx}",
                "__start_stamp__": transcript['start_stamp'],
                "__end_stamp__": transcript['end_stamp'],
                "__vector__": transcript_embeddings[idx]
            })

            transcript['index'] = f"t_{start_index + idx}"

        results = self._client.upsert(datas=list_data)
        return results
    
    async def query(self, query: str, top_k = 2, threshold = -1):
        embedder = imagebind_model.imagebind_huge(pretrained=True).cuda()
        embedder.eval()
        
        embedding = encode_string_query(query, embedder)
        embedding = embedding[0]
        results = self._client.query(
            query=embedding,
            top_k=top_k,
            better_than_threshold=threshold,
        )
        results = [
            {**dp, "id": dp["__id__"], "distance": dp["__metrics__"]} for dp in results
        ]
        return results
    
    async def index_done_callback(self):
        self._client.save()