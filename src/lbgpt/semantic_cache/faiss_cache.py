from pathlib import Path
from typing import Any, Optional

from langchain_community.docstore import InMemoryDocstore
from langchain_community.vectorstores.faiss import FAISS, dependable_faiss_import
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.embeddings import Embeddings
from openai.types.chat import ChatCompletion, CompletionCreateParams

from lbgpt.semantic_cache.base import _SemanticCacheBase, get_completion_create_params
from lbgpt.types import ChatCompletionAddition


class FaissSemanticCache(_SemanticCacheBase):
    def __init__(
        self,
        embedding_model: Embeddings,
        path: str | Path = "faiss_cache",
        cosine_similarity_threshold: float = 0.99,
    ):
        super().__init__(
            embedding_model=embedding_model,
            cosine_similarity_threshold=cosine_similarity_threshold,
        )

        # finding the FAISS model. Create it if it is not there
        if Path(path).exists():
            self.faiss_ = FAISS.load_local(
                str(path), embeddings=embedding_model, normalize_L2=True
            )

        else:
            faiss = dependable_faiss_import()
            # it is surprisingly difficult to figure out how long an embedding model is. Thus, we are actually
            # embedding a string here and then checking the length of the embedding
            _test_embedding = embedding_model.embed_query("test")

            self.faiss_ = FAISS(
                embedding_function=embedding_model,
                normalize_L2=True,
                index=faiss.IndexFlatL2(len(_test_embedding)),
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
                distance_strategy=DistanceStrategy.COSINE,
            )

    async def query_cache(
        self,
        query: CompletionCreateParams | dict[str, Any],
        semantic_cache_encoding_method: Optional[str],
    ) -> Optional[ChatCompletionAddition]:
        if semantic_cache_encoding_method is None:
            return None

        query = get_completion_create_params(**query)

        res = await self.faiss_.asimilarity_search_with_score_by_vector(
            embedding=self.embed_messages(
                query["messages"], encoding_method=semantic_cache_encoding_method
            ),
            filter=self.non_message_dict(query),
            k=1,
        )

        res = [r for r in res if r[1] <= 1 - self.cosine_similarity_threshold]

        if len(res) > 0:
            r = res[0][0].metadata["result"]
            r = {
                k: v
                for k, v in r.items()
                if k not in ["is_exact", "is_semantic_cached", "is_cached"]
            }

            return ChatCompletionAddition(
                **r, is_exact=res[0][1] <= 0.00001, is_semantic_cached=True
            )
        else:
            return

    async def add_cache(
        self,
        query: CompletionCreateParams | dict[str, Any],
        response: ChatCompletion,
        semantic_cache_encoding_method: Optional[str],
    ) -> None:
        if semantic_cache_encoding_method is None:
            return None

        query = get_completion_create_params(**query)
        text = self.encode_messages(
            query["messages"], encoding_method=semantic_cache_encoding_method
        )

        self.faiss_.add_embeddings(
            text_embeddings=[
                (
                    text,
                    self.embed_messages(
                        query["messages"],
                        encoding_method=semantic_cache_encoding_method,
                    ),
                )
            ],
            metadatas=[
                {
                    "result": response.model_dump(
                        exclude={
                            "is_exact",
                        }
                    ),
                    **self.non_message_dict(query),
                }
            ],
        )

    @property
    def count(self) -> int:
        return len(self.faiss_.index_to_docstore_id)
