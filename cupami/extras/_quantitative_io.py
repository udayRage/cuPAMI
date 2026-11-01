"""cuPAMI quantitative schemas: aligned uint32 IDs and float64 quantities.

BIN is a local versioned CSR format, not PAMI's binary format. Text follows
PAMI's items:quantities convention; Parquet uses Apache Arrow list columns.
https://arrow.apache.org/docs/python/parquet.html
"""
import csv
import math
import shutil
import struct
import tempfile
from collections.abc import Mapping
from itertools import islice
from numbers import Integral, Real
from pathlib import Path

import numpy as np

_HEADER = struct.Struct('<8sQQQ')
_MAGIC = b'CUPAMIQ1'


def normalize(row):
    if not isinstance(row, Mapping):
        raise ValueError('Quantitative rows must be mappings of item IDs to quantities')
    result = {}
    for label, quantity in row.items():
        if isinstance(label, bool) or not isinstance(label, (str, Integral)):
            raise ValueError('Quantitative item IDs must be uint32 integers')
        try:
            item = int(label)
        except ValueError:
            raise ValueError(f'Invalid item ID {label!r}') from None
        if not 0 <= item <= 0xFFFFFFFF or item in result:
            raise ValueError('Quantitative IDs must be unique uint32 integers')
        if isinstance(quantity, bool) or not isinstance(quantity, Real) or not math.isfinite(quantity) or quantity <= 0:
            raise ValueError('Quantities must be positive finite numbers')
        if isinstance(quantity, Integral) and quantity > 2**53:
            raise ValueError('Integer quantities must be at most 2**53 for exact float64 storage')
        result[item] = float(quantity)
    return result


def _arrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise ImportError('Parquet conversion requires pyarrow: pip install pyarrow') from error
    return pa, pq


def write(path, rows, sep=','):
    suffix = Path(path).suffix.lower()
    rows = (normalize(row) for row in rows)
    if suffix == '.bin':
        _write_binary(path, rows)
    elif suffix == '.parquet':
        pa, pq = _arrow()
        schema = pa.schema([
            pa.field('items', pa.list_(pa.field('item', pa.uint32(), nullable=False)), nullable=False),
            pa.field('quantities', pa.list_(pa.field('quantity', pa.float64(), nullable=False)), nullable=False),
        ])
        with pq.ParquetWriter(path, schema, compression='snappy') as writer:
            while batch := list(islice(rows, 8192)):
                writer.write_table(pa.Table.from_arrays([
                    pa.array([list(row) for row in batch], type=schema.field('items').type),
                    pa.array([list(row.values()) for row in batch], type=schema.field('quantities').type),
                ], schema=schema))
    elif suffix in ('', '.csv', '.tsv', '.txt', '.dat'):
        sep = '\t' if suffix == '.tsv' else sep
        with Path(path).open('w', encoding='utf-8', newline='') as output:
            for row in rows:
                ids = sep.join(map(str, row))
                values = sep.join(format(q, '.17g') for q in row.values())
                output.write(f'{ids}:{values}\n')
    else:
        raise ValueError('Use .csv, .tsv, .txt, .dat, .bin, or .parquet')


def _write_binary(path, rows):
    with tempfile.TemporaryFile() as offsets, tempfile.TemporaryFile() as ids, tempfile.TemporaryFile() as quantities:
        count = total = universe = 0
        offsets.write(struct.pack('<Q', 0))
        for row in rows:
            ids.write(struct.pack(f'<{len(row)}I', *row))
            quantities.write(struct.pack(f'<{len(row)}d', *row.values()))
            count += 1
            total += len(row)
            universe = max(universe, max(row, default=0))
            offsets.write(struct.pack('<Q', total))
        with Path(path).open('wb') as output:
            output.write(_HEADER.pack(_MAGIC, count, total, universe))
            for stream in (offsets, ids, quantities):
                stream.seek(0)
                shutil.copyfileobj(stream, output)


def _quantity(text):
    # Parse integer text before float conversion to detect inexact large ints.
    try:
        return int(text)
    except ValueError:
        return float(text)


def read(path, sep=None):
    suffix = Path(path).suffix.lower()
    if suffix == '.bin':
        return _read_binary(path)
    if suffix == '.parquet':
        pa, pq = _arrow()
        table = pq.read_table(path)
        if set(table.column_names) != {'items', 'quantities'}:
            raise ValueError('Quantitative Parquet requires items and quantities columns')
        ids, quantities = table['items'], table['quantities']
        if not pa.types.is_list(ids.type) or ids.type.value_type != pa.uint32() or not pa.types.is_list(quantities.type) or quantities.type.value_type != pa.float64():
            raise ValueError('Expected items: list<uint32> and quantities: list<float64>')
        if ids.null_count or quantities.null_count or ids.combine_chunks().flatten().null_count or quantities.combine_chunks().flatten().null_count:
            raise ValueError('Quantitative Parquet cannot contain nulls')
        return [_paired(items, values) for items, values in zip(ids.to_pylist(), quantities.to_pylist())]
    with Path(path).open(encoding='utf-8-sig') as source:
        rows = []
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            parts = line.strip().split(':')
            if len(parts) != 2:
                raise ValueError(f'Line {number}: expected items:quantities')
            delimiter = sep or ('\t' if '\t' in line else ',' if ',' in line else None)
            if delimiter:
                items, values = (next(csv.reader([part], delimiter=delimiter)) if part else [] for part in parts)
            else:
                items, values = (part.split() for part in parts)
            try:
                rows.append(_paired(items, [_quantity(q) for q in values]))
            except ValueError as error:
                raise ValueError(f'Line {number}: {error}') from error
        return rows


def _paired(items, quantities):
    if len(items) != len(quantities) or len(set(items)) != len(items):
        raise ValueError('Item and quantity lists must align and contain unique IDs')
    # Normalize before constructing the result to detect aliases such as 01/1.
    result = {}
    for label, quantity in zip(items, quantities):
        row = normalize({label: quantity})
        if next(iter(row)) in result:
            raise ValueError('Duplicate quantitative item ID')
        result.update(row)
    return result


def _read_binary(path):
    with Path(path).open('rb') as source:
        header = source.read(_HEADER.size)
        if len(header) != _HEADER.size:
            raise ValueError('Invalid quantitative BIN header')
        magic, count, total, universe = _HEADER.unpack(header)
        if magic != _MAGIC or universe > 0xFFFFFFFF:
            raise ValueError('Expected a quantitative BIN file (CUPAMIQ1)')
        if Path(path).stat().st_size != _HEADER.size + 8 * (count + 1) + 12 * total:
            raise ValueError('Invalid quantitative BIN size')
        offsets = np.fromfile(source, dtype='<u8', count=count + 1)
        items = np.fromfile(source, dtype='<u4', count=total)
        quantities = np.fromfile(source, dtype='<f8', count=total)
    if len(offsets) != count + 1 or len(items) != total or len(quantities) != total:
        raise ValueError('Truncated quantitative BIN arrays')
    if offsets[0] != 0 or offsets[-1] != total or np.any(offsets[1:] < offsets[:-1]):
        raise ValueError('Invalid quantitative BIN offsets')
    if np.any(items > universe):
        raise ValueError('Quantitative item ID exceeds header bound')
    return [_paired(items[int(a):int(b)].tolist(), quantities[int(a):int(b)].tolist())
            for a, b in zip(offsets, offsets[1:])]
