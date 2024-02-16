import uuid
from types import NoneType
from typing import Any, Optional
from langchain_core.embeddings import Embeddings
from openai.types.chat import ChatCompletion, CompletionCreateParams
from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException
from qdrant_client.http.models import FieldCondition, Filter, MatchValue, PointStruct
from qdrant_client.models import Distance, VectorParams
from lbgpt.semantic_cache.base import _SemanticCacheBase, get_completion_create_params
from lbgpt.types import ChatCompletionAddition


from logging import getLogger


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
        self, query: CompletionCreateParams | dict[str, Any]
    ) -> Optional[ChatCompletionAddition]:
        try:
            return await self._query_cache(query=query)
        except ResponseHandlingException as e:
            logger.debug(f"Error querying cache, proceed anyways: {e}")
            # we don't really care about errors here
            return

    async def add_cache(
        self, query: CompletionCreateParams | dict[str, Any], response: ChatCompletion
    ) -> None:
        try:
            return await self._add_cache(query=query, response=response)
        except ResponseHandlingException as e:
            logger.debug(f"Error querying cache, proceed anyways: {e}")
            # we don't really care about errors here
            return

    async def _query_cache(
        self, query: CompletionCreateParams | dict[str, Any]
    ) -> Optional[ChatCompletionAddition]:
        query = get_completion_create_params(**query)

        filter_params = self.non_message_dict(query)

        query_filter = Filter(
            must=[
                FieldCondition(
                    key=k, match=MatchValue(value=self._create_filter_value(v))
                )
                for k, v in filter_params.items()
                if isinstance(v, self.ALLOWED_TYPES_FOR_PAYLOAD) or v is None
            ]
        )

        embedding = await self.aembed_messages(query["messages"])

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
            return ChatCompletionAddition(
                **res[0].payload["result"],
                is_exact=res[0].score >= 1 - 0.00001,
            )
        else:
            return

    async def _add_cache(
        self, query: CompletionCreateParams | dict[str, Any], response: ChatCompletion
    ) -> None:
        query = get_completion_create_params(**query)
        embedding = await self.aembed_messages(query["messages"])

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
