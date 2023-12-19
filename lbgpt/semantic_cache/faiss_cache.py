from pathlib import Path
from typing import Any, Optional

from langchain.docstore import InMemoryDocstore
from langchain.vectorstores.faiss import FAISS, dependable_faiss_import
from langchain.vectorstores.utils import DistanceStrategy
from langchain_core.embeddings import Embeddings
from openai.types.chat import CompletionCreateParams, ChatCompletion

from lbgpt.semantic_cache.base import _SemanticCacheBase
from lbgpt.types import ChatCompletionAddition


class FaissSemanticCache(_SemanticCacheBase):
    def __init__(
        self,
        embedding_model: Embeddings,
        cosine_similarity_threshold: float = 0.99,
        path: str | Path = "faiss_cache",
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
                distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
            )

        self._euclidean_threshold = 2 - 2 * cosine_similarity_threshold

    def query_cache(self, query: CompletionCreateParams | dict[str, Any]) -> Optional[ChatCompletionAddition]:
        query = CompletionCreateParams(**query)

        res = self.faiss_.similarity_search_with_score_by_vector(
            embedding=self.embed_messages(query["messages"]),
            filter=self.non_message_dict(query),
            k=1,
            threshold=self._euclidean_threshold,
        )

        if len(res) > 0:
            return ChatCompletionAddition(**res[0][0].metadata["result"], is_exact=res[0][1] >= 1.0)
        else:
            return

    def add_cache(
        self, query: CompletionCreateParams | dict[str, Any], response: ChatCompletion
    ) -> None:
        query = CompletionCreateParams(**query)

        self.faiss_.add_embeddings(
            text_embeddings=[
                (
                    self.messages_to_text(query["messages"]),
                    self.embed_messages(query["messages"]),
                )
            ],
            metadatas=[{"result": response, **self.non_message_dict(query)}],
        )
