import time
import asyncio
import requests
import httpx
from typing import List, Dict
from pydantic import BaseModel, Field
from langchain_core.embeddings import Embeddings
import logging

# Configure logging
logger = logging.getLogger("BoeingEmbeddings")
logger.setLevel(logging.INFO)


class BoeingEmbeddings(Embeddings, BaseModel):
    """
    LangChain-compatible wrapper for Boeing API with Batching and Retries.
    """
    # --- Configuration Fields (from attached file) ---
    api_url: str = Field(default="https://bcai-test.web.boeing.com/bcai-public-api/embedding", description="The API endpoint for embeddings.")
    udal_pat: str = Field(..., description="The UDAL_PAT token.")
    model: str = Field(default="text-embedding-3-large", description="Model name.")
    batch_size: int = Field(default=250, description="Batch size for API calls.")
    max_retries: int = Field(default=5, description="Max retries for failures.")

    def _get_headers(self) -> Dict[str, str]:
        return {
            'accept': 'application/json',
            'Content-Type': 'application/json',
            'Authorization': f'basic {self.udal_pat}'}

    def _embed_batch(self, texts: List[str]) -> List[List[float]] | List[None]:
        """Internal helper to handle the actual API call with retries """
        payload = {"input": texts, "model": self.model}

        for attempt in range(self.max_retries):
            try:
                response = requests.post(self.api_url, headers=self._get_headers(), json=payload)
                if response.status_code == 200:
                    data = response.json().get("data", [])
                    # RE-SORTING LOGIC
                    sorted_embeddings = [None] * len(texts)
                    for item in data:
                        idx = item.get('index')
                        if idx is not None and 0 <= idx < len(sorted_embeddings):
                            sorted_embeddings[idx] = item.get('embedding')

                    if None in sorted_embeddings:
                        raise ValueError("Missing embeddings in response")
                    return sorted_embeddings
                else:
                    logger.warning(f"Attempt {attempt + 1} failed: {response.status_code}")
            except Exception as e:
                logger.error(f"Error: {e}")

            time.sleep(2 ** attempt)  # Exponential backoff

        raise RuntimeError(f"Failed to embed batch after {self.max_retries} attempts")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embeds a list of documents using batching."""
        all_embeddings = []
        # Batching logic
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self._embed_batch(batch)
            all_embeddings.extend(batch_embeddings)
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embeds a single query."""
        return self._embed_batch([text])[0]  # Reuse batch logic for consistency

    async def _async_embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Async equivalent of _embed_batch using httpx.AsyncClient.

        Natively async — the event loop is free while awaiting the BCAI
        API response, allowing concurrent coroutines (other file parsing,
        SQLite writes) to run in parallel.

        Retry behaviour, exponential backoff, and re-sorting logic are
        identical to the sync version to guarantee consistent results
        across both ingestion and retrieval paths.

        Args:
            texts: Batch of text strings to embed in a single API call.

        Returns:
            List of embedding vectors sorted by original input index.

        Raises:
            RuntimeError: If all retry attempts are exhausted.
        """
        payload = {"input": texts, "model": self.model}

        for attempt in range(self.max_retries):
            try:
                # New client per attempt — avoids stale connection reuse
                # after a failed attempt. verify=False required for Boeing
                # internal TLS certificates.
                async with httpx.AsyncClient(verify=False, timeout=60.0) as client:
                    response = await client.post(
                        self.api_url,
                        headers=self._get_headers(),
                        json=payload,
                    )

                if response.status_code == 200:
                    data = response.json().get("data", [])

                    # Re-sort by index — BCAI does not guarantee response order.
                    sorted_embeddings: List = [None] * len(texts)
                    for item in data:
                        idx = item.get("index")
                        if idx is not None and 0 <= idx < len(sorted_embeddings):
                            sorted_embeddings[idx] = item.get("embedding")

                    if None in sorted_embeddings:
                        raise ValueError("Missing embeddings in response — index gap detected.")

                    return sorted_embeddings

                logger.warning(
                    f"Async attempt {attempt + 1}/{self.max_retries} failed: "
                    f"HTTP {response.status_code}"
                )

            except Exception as e:
                logger.error(f"Async embedding error on attempt {attempt + 1}: {e}")

            # Async-safe exponential backoff — does not block the event loop.
            await asyncio.sleep(2 ** attempt)

        raise RuntimeError(
            f"Async embedding failed after {self.max_retries} attempts."
        )

    async def async_embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Async batched embedding for document ingestion.

        Called by IngestionPipeline.ingest_file() during Phase A of the
        two-phase upsert. Pre-computing embeddings here (async, network I/O)
        before passing them to ChromaDB.upsert(embeddings=...) means ChromaDB
        never invokes its internal sync embedding function during ingestion.

        The sync embed_documents() and embed_query() are intentionally
        preserved and remain active for the VectorDBManager retrieval path,
        where ChromaDB calls them internally via collection.query().

        Args:
            texts: Full list of document text strings to embed.

        Returns:
            Flat list of embedding vectors, one per input text, in order.
        """
        all_embeddings: List[List[float]] = []

        for batch_num, i in enumerate(range(0, len(texts), self.batch_size), start=1):
            batch = texts[i: i + self.batch_size]
            batch_embeddings = await self._async_embed_batch(batch)
            all_embeddings.extend(batch_embeddings)

            logger.info(
                f"Async embedded batch {batch_num} "
                f"({len(batch)} texts | {len(all_embeddings)}/{len(texts)} total)"
            )

        return all_embeddings

