import uuid
from logging import getLogger
from types import NoneType
from typing import Any, Optional

from langchain_core.embeddings import Embeddings
from openai.types.chat import ChatCompletion, CompletionCreateParams
from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException
from qdrant_client.http.models import (
    FieldCondition,
    Filter,
    MatchValue,
    PayloadFieldSchema,
    PayloadSchemaType,
    PointStruct,
)
from qdrant_client.models import Distance, VectorParams

from lbgpt.cache import make_hash_chatgpt_request
from lbgpt.semantic_cache.base import _SemanticCacheBase, get_completion_create_params
from lbgpt.types import ChatCompletionAddition

logger = getLogger(__name__)


class QdrantClientConfig(BaseModel):
    host: Optional[str] = None
    port: Optional[int] = 6333
    url: Optional[str] = None
    api_key: Optional[str] = None

    class Config:
        extra = "allow"

    def get_async_client(self) -> AsyncQdrantClient:
        if self.api_key == "":
            api_key = None
        else:
            api_key = self.api_key

        return AsyncQdrantClient(
            host=self.host,
            port=self.port,
            url=self.url,
            api_key=api_key,
            **self.extra_fields,
        )

    def get_sync_client(self) -> QdrantClient:
        if self.api_key == "":
            api_key = None
        else:
            api_key = self.api_key
        return QdrantClient(
            host=self.host,
            port=self.port,
            url=self.url,
            api_key=api_key,
            **self.extra_fields,
        )

    @property
    def extra_fields(self) -> dict[str, Any]:
        field_names = set(self.__dict__) - set(self.__fields__)
        return {fn: getattr(self, fn) for fn in field_names}


class QdrantSemanticCache(_SemanticCacheBase):
    ALLOWED_TYPES_FOR_PAYLOAD = (int, str, bool)

    def __init__(
        self,
        embedding_model: Embeddings,
        collection_name: str,
        host: Optional[str] = None,
        port: Optional[int] = 6333,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        qdrant_properties: Optional[dict[str, Any]] = None,
        cosine_similarity_threshold: float = 0.99,
        max_concurrent_requests=10,
    ):
        super().__init__(
            embedding_model=embedding_model,
            cosine_similarity_threshold=cosine_similarity_threshold,
        )

        # setting up the Qdrant client properly
        # we are setting up a synchroneous and an asynchroneous client
        # the synchroneous client is used for special operations like counting, listing, where we actually
        # care about precision

        self.collection_name = collection_name
        self._QdrantClientConfig = QdrantClientConfig(
            host=host, port=port, url=url, api_key=api_key, **(qdrant_properties or {})
        )

        client = self._QdrantClientConfig.get_sync_client()
        collection_result = client.get_collections()

        existing_collection_names = [k.name for k in collection_result.collections]
        if collection_name not in existing_collection_names:
            # it is surprisingly difficult to figure out how long an embedding model is. Thus, we are actually
            # embedding a string here and then checking the length of the embedding
            _test_embedding = embedding_model.embed_query("test")

            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=len(_test_embedding), distance=Distance.COSINE
                ),
            )

            client.create_payload_index(
                collection_name=self.collection_name,
                field_name="hashed_model",
                field_schema=PayloadSchemaType.TEXT,
            )

        client.close()

    def _create_filter_value(
        self, value: int | str | bool | NoneType
    ) -> int | str | bool:
        if isinstance(value, self.ALLOWED_TYPES_FOR_PAYLOAD):
            return value
        elif value is None:
            return ""
        else:
            raise NotImplementedError(f"cannot process type {type(value)}")

    async def query_cache(
        self,
        query: CompletionCreateParams | dict[str, Any],
        semantic_cache_encoding_method: Optional[str],
    ) -> Optional[ChatCompletionAddition]:
        if semantic_cache_encoding_method is None:
            return

        try:
            return await self._query_cache(
                query=query,
                semantic_cache_encoding_method=semantic_cache_encoding_method,
            )
        except ResponseHandlingException as e:
            logger.debug(f"Error querying cache, proceed anyways: {e}")
            # we don't really care about errors here
            return

    async def add_cache(
        self,
        query: CompletionCreateParams | dict[str, Any],
        response: ChatCompletion,
        semantic_cache_encoding_method: Optional[str],
    ) -> None:
        if semantic_cache_encoding_method is None:
            return

        try:
            return await self._add_cache(
                query=query,
                response=response,
                semantic_cache_encoding_method=semantic_cache_encoding_method,
            )
        except ResponseHandlingException as e:
            logger.debug(f"Error querying cache, proceed anyways: {e}")
            # we don't really care about errors here
            return

    def hashed_query_payload(
        self, query: CompletionCreateParams | dict[str, Any]
    ) -> str:
        return make_hash_chatgpt_request(query, include_messages=False)

    async def _query_cache(
        self,
        query: CompletionCreateParams | dict[str, Any],
        semantic_cache_encoding_method: str,
    ) -> Optional[ChatCompletionAddition]:
        query = get_completion_create_params(**query)
        filter_params_hash = make_hash_chatgpt_request(query, include_messages=False)

        query_filter = Filter(
            must=[
                FieldCondition(
                    key="hashed_model", match=MatchValue(value=filter_params_hash)
                )
            ]
        )

        embedding = await self.aembed_messages(
            query["messages"], encoding_method=semantic_cache_encoding_method
        )

        res = await self._QdrantClientConfig.get_async_client().search(
            collection_name=self.collection_name,
            query_vector=embedding,
            query_filter=query_filter,
            with_payload=True,
            with_vectors=False,
            limit=1,
            score_threshold=self.cosine_similarity_threshold,
            timeout=10,
        )

        res = [r for r in res if r.score >= self.cosine_similarity_threshold]

        if len(res) > 0:
            r = res[0].payload["result"]
            r = {
                k: v
                for k, v in r.items()
                if k not in ["is_exact", "is_semantic_cached", "is_cached"]
            }

            return ChatCompletionAddition(
                **r,
                is_exact=res[0].score >= 1 - 0.00001,
                is_semantic_cached=True,
            )
        else:
            return

    async def _add_cache(
        self,
        query: CompletionCreateParams | dict[str, Any],
        response: ChatCompletion,
        semantic_cache_encoding_method: str,
    ) -> None:
        query = get_completion_create_params(**query)
        filter_params_hash = make_hash_chatgpt_request(query, include_messages=False)

        embedding = await self.aembed_messages(
            query["messages"], encoding_method=semantic_cache_encoding_method
        )

        await self._QdrantClientConfig.get_async_client().upsert(
            collection_name=self.collection_name,
            wait=False,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "result": response.model_dump(
                            exclude={
                                "is_exact",
                            }
                        ),
                        "hashed_model": filter_params_hash,
                        **self.non_message_dict(
                            query,
                            allowed_types=self.ALLOWED_TYPES_FOR_PAYLOAD,
                            convert_not_allowed_to_empty=True,
                        ),
                    },
                )
            ],
        )

    @property
    def count(self) -> int:
        return (
            self._QdrantClientConfig.get_sync_client()
            .count(collection_name=self.collection_name)
            .count
        )
