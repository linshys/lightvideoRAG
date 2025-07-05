import os
import shutil
import asyncio
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Type, Union, cast

from ._videoutil import(
    similarity_sampling,
    similarity_sampling_kornia_gpu,
    embedder_image_bind_frames,
    embedder_image_bind_transcript,
    load_transcripts
)

from ._retriever import (
    lightvideorag_query
)
from ._storage import (
    InsertTaskManager,
    JsonKVStorage,
    NanoVectorDBStorage,
    NanoVectorDBVideoSegmentStorage
)
from ._utils import (
    convert_response_to_json,
    always_get_an_event_loop,
    logger,
)
from .base import (
    BaseKVStorage,
    BaseVectorStorage,
    StorageNameSpace,
    QueryParam
)

from ._vlm import ovis2_unload

@dataclass
class LightVideoRAG:
    working_dir: str = field(
        default_factory=lambda: f"./lightvideorag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )
    
    # video
    threads_for_split: int = 10
    video_segment_length: int = 30 # seconds
    rough_num_frames_per_segment: int = 5 # frames
    fine_num_frames_per_segment: int = 15 # frames
    video_output_format: str = "mp4"
    audio_output_format: str = "wav"
    video_embedding_batch_num: int = 2
    segment_retrieval_top_k: int = 4
    video_embedding_dim: int = 1024
    
    # query
    retrieval_topk_chunks: int = 2
    
    # storage
    key_string_value_json_storage_cls: Type[BaseKVStorage] = JsonKVStorage
    vector_db_storage_cls: Type[BaseVectorStorage] = NanoVectorDBStorage
    vs_vector_db_storage_cls: Type[BaseVectorStorage] = NanoVectorDBVideoSegmentStorage
    vector_db_storage_cls_kwargs: dict = field(default_factory=dict)
    enable_llm_cache: bool = True

    # extension
    always_create_working_dir: bool = True
    addon_params: dict = field(default_factory=dict)
    convert_response_to_json_func: callable = convert_response_to_json
    
    def __post_init__(self):
        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self).items()])
        logger.debug(f"LightVideoRAG init with param:\n\n  {_print_config}\n")

        if not os.path.exists(self.working_dir) and self.always_create_working_dir:
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)

        self.video_path_db = self.key_string_value_json_storage_cls(
            namespace="video_path", global_config=asdict(self)
        )

        self.video_frames_feature_vdb = (
            self.vs_vector_db_storage_cls(
                namespace="video_frames_feature",
                global_config=asdict(self),
                embedding_func=None, # we code the embedding process inside the insert() function.
            )
        )

        self.video_transcript_feature_vdb = (
            self.vs_vector_db_storage_cls(
                namespace="video_transcript_feature",
                global_config=asdict(self),
                embedding_func=None, # we code the embedding process inside the insert() function.
            )
        )

        ## both feature in one vdb
        self.video_frames_transcript_vdb = (
            self.vs_vector_db_storage_cls(
                namespace="video_frames_transcript",
                global_config=asdict(self),
                embedding_func=None, # we code the embedding process inside the insert() function.
            )
        )

        ## kv data for transcript
        self.transcripts_meta_kv = self.key_string_value_json_storage_cls(
            namespace="transcripts_meta", global_config=asdict(self)
        )


    def insert_video(self, video_path_list=None, extra: dict = None):
        loop = always_get_an_event_loop()
        for video_path in video_path_list:
            # Step0: check the existence
            video_name = os.path.basename(video_path).split('.')[0]
            if video_name in self.video_path_db._data:
                logger.info(f"Find the video named {os.path.basename(video_path)} in storage and skip it.")
                continue
            loop.run_until_complete(self.video_path_db.upsert(
                {video_name: video_path}
            ))

            task_mgr = InsertTaskManager()

            # Step1: apply sampling to video
            timestamp,frames_tensor = similarity_sampling_kornia_gpu(video_path)

            # Step2: encode frames to embeddings
            frame_embeddings = embedder_image_bind_frames(frames_tensor)

            # Step3: upsertVdb
            task_mgr.submit(self.video_frames_feature_vdb.upsert_frames(video_name, frame_embeddings, timestamp))
            # task_mgr.submit(self.video_frames_transcript_vdb.upsert_frames(video_name, frame_embeddings, timestamp))

            # Step4: obtain transcript with whisper or load from json file
            transcripts = load_transcripts(
                video_path,
                self.audio_output_format,
                extra = extra
            )

            # Step5: encode text to embeddings
            transcript_embeddings = embedder_image_bind_transcript(transcripts)

            # Step6: upsert transcript Vdb
            # task_mgr.submit(self.video_frames_transcript_vdb.upsert_transcripts(
            #         video_name, transcripts, transcript_embeddings))
            task_mgr.submit(self.video_transcript_feature_vdb.upsert_transcripts(
                    video_name, transcripts, transcript_embeddings))

            loop.run_until_complete(task_mgr.wait_all())

            # Step7: save transcript kv
            transcripts = {item["index"]: item for item in transcripts}
            loop.run_until_complete(self.transcripts_meta_kv.upsert(
                {video_name: transcripts}
            ))

            # Step 8: delete the cache file
            video_segment_cache_path = os.path.join(self.working_dir, '_cache', video_name)
            if os.path.exists(video_segment_cache_path):
                shutil.rmtree(video_segment_cache_path)
            #
            # Step 9: saving current video information
            loop.run_until_complete(self._save_video_segments())


    def query(self, query: str, param: QueryParam = QueryParam()):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery(query, param))

    async def aquery(self, query: str, param: QueryParam = QueryParam()):
        if param.mode == "videorag":
            response = await lightvideorag_query(
                query,
                self.video_path_db,
                self.video_frames_feature_vdb,
                self.video_transcript_feature_vdb,
                self.transcripts_meta_kv,
                self.video_frames_transcript_vdb,
                param,
                asdict(self),
            )
        else:
            raise ValueError(f"Unknown mode {param.mode}")
        return response

    async def _save_video_segments(self):
        tasks = []
        for storage_inst in [
            self.video_frames_feature_vdb,
            self.video_transcript_feature_vdb,
            self.video_frames_transcript_vdb,
            self.transcripts_meta_kv,
            self.video_path_db,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    def unload_model(self):
        ovis2_unload()