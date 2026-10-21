"""Temporal BIN and Parquet storage with timestamps separate from item IDs."""

import shutil
import struct
import tempfile
from itertools import islice
from numbers import Integral
from pathlib import Path

import numpy as np

_MAGIC = b"CUPAMIT1"
_HEADER = struct.Struct("<8sQQQ")


def _row(row, universe):
    timestamp, *items = row
    if isinstance(timestamp, bool) or not isinstance(timestamp, Integral) or not 0 <= timestamp <= 0x7FFFFFFFFFFFFFFF:
        raise ValueError("Temporal timestamps must be nonnegative int64 integers")
    if any(isinstance(item, bool) or not isinstance(item, Integral) or not 0 <= item <= universe for item in items):
        raise ValueError("Temporal item IDs must be integers within the configured item range")
    return int(timestamp), list(map(int, items))


def write_binary(path, rows, universe):
    if not 1 <= universe <= 0xFFFFFFFF:
        raise ValueError("BIN item IDs must fit in uint32")
    with tempfile.TemporaryFile() as timestamps, tempfile.TemporaryFile() as offsets, tempfile.TemporaryFile() as items:
        count = total = 0
        offsets.write(struct.pack("<Q", 0))
        for row in rows:
            timestamp, basket = _row(row, universe)
            timestamps.write(struct.pack("<q", timestamp))
            items.write(struct.pack(f"<{len(basket)}I", *basket))
            count += 1
            total += len(basket)
            offsets.write(struct.pack("<Q", total))
        with Path(path).open("wb") as output:
            output.write(_HEADER.pack(_MAGIC, count, total, universe))
            for stream in (timestamps, offsets, items):
                stream.seek(0)
                shutil.copyfileobj(stream, output)


def read_binary(path):
    with Path(path).open("rb") as source:
        header = source.read(_HEADER.size)
        if len(header) != _HEADER.size:
            raise ValueError("Invalid temporal BIN header")
        magic, count, total, universe = _HEADER.unpack(header)
        if magic != _MAGIC:
            raise ValueError("Expected a temporal BIN file with timestamp data")
        if not 1 <= universe <= 0xFFFFFFFF:
            raise ValueError("Invalid temporal BIN item range")
        if Path(path).stat().st_size != _HEADER.size + 8 * count + 8 * (count + 1) + 4 * total:
            raise ValueError("Invalid temporal BIN file size")
        timestamps = np.fromfile(source, dtype="<i8", count=count)
        offsets = np.fromfile(source, dtype="<u8", count=count + 1)
        items = np.fromfile(source, dtype="<u4", count=total)
    if len(timestamps) != count or len(offsets) != count + 1 or len(items) != total:
        raise ValueError("Truncated temporal BIN arrays")
    if np.any(timestamps < 0):
        raise ValueError("Temporal timestamps must be nonnegative")
    if offsets[0] != 0 or offsets[-1] != total or np.any(offsets[1:] < offsets[:-1]):
        raise ValueError("Invalid temporal BIN offsets")
    if np.any(items > universe):
        raise ValueError("Temporal BIN item ID exceeds header range")
    return [[int(timestamp), *items[int(offsets[i]):int(offsets[i + 1])].tolist()]
            for i, timestamp in enumerate(timestamps)]


def _arrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise ImportError("Temporal Parquet requires pyarrow: pip install pyarrow") from error
    return pa, pq


def write_parquet(path, rows, universe):
    if not 1 <= universe <= 0xFFFFFFFF:
        raise ValueError("Parquet item IDs must fit in uint32")
    pa, pq = _arrow()
    schema = pa.schema([
        pa.field("timestamp", pa.int64(), nullable=False),
        pa.field("items", pa.list_(pa.field("item", pa.uint32(), nullable=False)), nullable=False),
    ])
    rows = iter(rows)
    with pq.ParquetWriter(path, schema, compression="snappy") as writer:
        while True:
            batch = [_row(row, universe) for row in islice(rows, 8192)]
            if not batch:
                break
            table = pa.Table.from_arrays([
                pa.array([row[0] for row in batch], type=pa.int64()),
                pa.array([row[1] for row in batch], type=schema.field("items").type),
            ], schema=schema)
            writer.write_table(table)


def read_parquet(path):
    pa, pq = _arrow()
    table = pq.read_table(path)
    if "timestamp" not in table.column_names or "items" not in table.column_names:
        raise ValueError("Temporal Parquet requires timestamp and items columns")
    timestamp, items = table["timestamp"].combine_chunks(), table["items"].combine_chunks()
    if timestamp.type != pa.int64() or not pa.types.is_list(items.type) or items.type.value_type != pa.uint32():
        raise ValueError("Temporal Parquet requires timestamp: int64 and items: list<uint32>")
    if timestamp.null_count or items.null_count or items.flatten().null_count:
        raise ValueError("Temporal Parquet cannot contain null timestamps, baskets, or items")
    timestamps = timestamp.to_pylist()
    if any(value < 0 for value in timestamps):
        raise ValueError("Temporal timestamps must be nonnegative")
    return [[ts, *basket] for ts, basket in zip(timestamps, items.to_pylist())]
