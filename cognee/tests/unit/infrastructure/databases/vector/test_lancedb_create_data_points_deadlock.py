"""
Regression test for issue #5987: LanceDBAdapter.create_data_points deadlocks
when the collection does not exist yet.

The deadlock occurred because create_data_points acquired VECTOR_DB_LOCK and
then called create_collection, which tried to acquire the same non-reentrant
lock again.
"""

from __future__ import annotations

import asyncio
import hashlib

import pytest

try:
    from cognee.infrastructure.databases.vector.lancedb.LanceDBAdapter import LanceDBAdapter
    from cognee.infrastructure.engine import DataPoint

    HAS_LANCEDB = True
except ModuleNotFoundError:
    HAS_LANCEDB = False


class _HashEmbeddingEngine:
    """Deterministic embedding engine that needs no API key or network."""

    async def embed_text(self, texts):
        return [[b / 255.0 for b in hashlib.sha256(t.encode()).digest()[:16]] for t in texts]

    def get_vector_size(self):
        return 16

    def get_batch_size(self):
        return 32

    async def input_limit(self):
        return None


class _Note(DataPoint):
    """Test DataPoint with a text field."""

    text: str
    metadata: dict = {"index_fields": ["text"]}


def _make_adapter(tmp_path) -> LanceDBAdapter:
    return LanceDBAdapter(
        url=str(tmp_path / "db"),
        api_key=None,
        embedding_engine=_HashEmbeddingEngine(),
    )


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb not installed")
async def test_create_data_points_on_new_collection_does_not_deadlock(tmp_path):
    """
    Regression: create_data_points on a new collection must not hang forever.

    Before the fix, create_data_points acquired VECTOR_DB_LOCK and then called
    create_collection, which tried to acquire the same lock again, causing a
    deadlock on the same event loop.
    """
    adapter = _make_adapter(tmp_path)
    collection_name = "notes"

    # This must complete without timing out. The timeout is generous (20s) to
    # accommodate slow CI runners while still detecting the deadlock, which
    # would hang indefinitely.
    try:
        await asyncio.wait_for(
            adapter.create_data_points(collection_name, [_Note(text="x")]), timeout=20.0
        )
    except asyncio.TimeoutError:
        pytest.fail("create_data_points on new collection deadlocked (timeout after 20s)")

    # Verify the data point was actually created
    collection = await adapter.get_collection(collection_name)
    table = await collection.to_arrow()
    rows = table.to_pylist()
    assert len(rows) == 1
    assert rows[0]["payload"]["text"] == "x"


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb not installed")
async def test_create_data_points_on_existing_collection_still_works(tmp_path):
    """
    Sanity check: create_data_points on an existing collection must still work.

    This case never had the deadlock because the outer if-check short-circuits
    when the collection already exists, but we verify the fix doesn't break it.
    """
    adapter = _make_adapter(tmp_path)
    collection_name = "notes"

    # Create collection first
    await adapter.create_collection(collection_name, _Note)

    # Now add data points — this path never had the deadlock
    await asyncio.wait_for(
        adapter.create_data_points(collection_name, [_Note(text="y")]), timeout=20.0
    )

    collection = await adapter.get_collection(collection_name)
    table = await collection.to_arrow()
    rows = table.to_pylist()
    assert len(rows) == 1
    assert rows[0]["payload"]["text"] == "y"
