"""Convert database files through one interface selected by data type.

Local cuPAMI conversion routines; schemas: docs/transaction-formats.md.
Parquet list-column reference: https://arrow.apache.org/docs/python/parquet.html
CLI: python -m cupami.extras.datasetConverter --help
"""

from pathlib import Path


def convertFile(source, outputFile, dataType="transactional", sep=None, *, outputSep=None):
    """Convert CSV/TSV/text, BIN, or Parquet in either direction without CUDA.

    dataType is binary (or transactional), temporal, or quantitative. sep is
    the input text delimiter; outputSep overrides the output text delimiter.
    Output is written atomically so validation errors preserve existing files.
    Quantitative input rows are mappings; file rows use items:quantities.
    """
    handlers = {"transactional": _convertTransactional, "binary": _convertTransactional,
                "temporal": _convertTemporal, "quantitative": _convertQuantitative}
    if not isinstance(dataType, str) or dataType.lower() not in handlers:
        raise ValueError(f"Unsupported data type {dataType!r}; supported: binary, transactional, temporal, quantitative")
    path = Path(outputFile)
    suffix = path.suffix.lower()
    if suffix not in (".csv", ".tsv", ".txt", ".dat", "", ".bin", ".parquet"):
        raise ValueError("Use .csv, .tsv, .txt, .dat, .bin, or .parquet output")
    if outputSep is not None and (not isinstance(outputSep, str) or len(outputSep) != 1 or outputSep in '\r\n:"'):
        raise ValueError('outputSep must be one character other than newline, colon, or quote')
    if outputSep is not None and suffix in ('.bin', '.parquet'):
        raise ValueError('outputSep applies only to text output')
    import os
    import tempfile
    descriptor, temporary = tempfile.mkstemp(prefix='.' + path.stem + '-', suffix=suffix, dir=path.parent)
    os.close(descriptor)
    try:
        summary = handlers[dataType.lower()](source, temporary, sep=sep)
        if outputSep is not None:
            import csv
            delimiter = '\t' if suffix == '.tsv' else ','
            temp_path = Path(temporary)
            if dataType.lower() == 'quantitative':
                from . import _quantitative_io
                rows = _quantitative_io.read(temp_path, sep=delimiter)
                # Use the selected delimiter even when the extension is TSV.
                text = ''.join(outputSep.join(map(str, row)) + ':' +
                               outputSep.join(format(q, '.17g') for q in row.values()) + '\n' for row in rows)
                temp_path.write_text(text, encoding='utf-8')
            else:
                with temp_path.open(newline='', encoding='utf-8') as stream:
                    rows = list(csv.reader(stream, delimiter=delimiter))
                with temp_path.open('w', newline='', encoding='utf-8') as stream:
                    csv.writer(stream, delimiter=outputSep).writerows(rows)
        os.replace(temporary, path)
        return summary
    finally:
        Path(temporary).unlink(missing_ok=True)


def convertTransactions(source, outputFile, sep=None):
    """Compatibility shorthand for convertFile(..., dataType='transactional')."""
    return convertFile(source, outputFile, dataType="transactional", sep=sep)


def _convertTransactional(source, outputFile, sep=None):
    """Convert local CSV/BIN/Parquet, delimited text URLs, or transaction rows.

    Output format is selected by extension. Item IDs must be integers in
    0..2**32-1. Repeated items are removed and empty baskets are omitted,
    matching cuApriori's input semantics. This utility does not require CUDA;
    Parquet requires PyArrow. The database is retained in CPU memory.
    """
    from cupami.frequentPattern import cuApriori
    from .syntheticDataGenerator import TransactionalDatabase

    suffix = Path(outputFile).suffix.lower()
    if suffix not in (".csv", ".bin", ".parquet", ".tsv", ".txt", ".dat", ""):
        raise ValueError("Use a .csv, .bin, or .parquet output file")
    reader = cuApriori(source, 1, sep=sep, use_gds=False, use_gpu_io=False)
    transactions = []
    universe = 0
    occurrences = 0
    for rowIndex, row in enumerate(reader._Database, start=1):
        transaction = []
        for label in row:
            try:
                item = int(label)
            except ValueError as error:
                raise ValueError(
                    f"Transaction {rowIndex}: item {label!r} is not an integer ID"
                ) from error
            if not 0 <= item <= 0xFFFFFFFF:
                raise ValueError(f"Transaction {rowIndex}: item IDs must be in 0..2**32-1")
            if item not in transaction:
                transaction.append(item)
        universe = max(universe, max(transaction))
        occurrences += len(transaction)
        transactions.append(transaction)

    # Reuse the generator's format writer so the schemas cannot drift.
    writer = TransactionalDatabase(len(transactions), max(1, universe))
    writer._writeDatabase(outputFile, iter(transactions))
    return {"transactions": len(transactions), "item_occurrences": occurrences,
            "max_item_id": universe if transactions else None,
            "format": suffix.lstrip(".") if suffix in (".bin", ".parquet") else "tsv" if suffix == ".tsv" else "csv"}


def _convertTemporal(source, outputFile, sep=None):
    """Convert temporal files while preserving timestamps and separate rows."""
    from ..periodicFrequentPattern._temporal import TemporalInput
    from .syntheticDataGenerator import TemporalDatabase

    suffix = Path(outputFile).suffix.lower()
    if suffix not in (".csv", ".bin", ".parquet", ".tsv", ".txt", ".dat", ""):
        raise ValueError("Use a .csv, .bin, or .parquet output file")
    reader = TemporalInput(source, minSup=1, maxPer=0, sep=sep)
    rows = []
    universe = occurrences = 0
    for row_index, (timestamp, basket) in enumerate(zip(reader._timestamps, reader._Database), start=1):
        items = []
        for label in basket:
            try:
                item = int(label)
            except ValueError as error:
                raise ValueError(f"Transaction {row_index}: item {label!r} is not an integer ID") from error
            if not 0 <= item <= 0xFFFFFFFF:
                raise ValueError(f"Transaction {row_index}: item IDs must be in 0..2**32-1")
            if item not in items:
                items.append(item)
        universe = max(universe, max(items, default=0))
        occurrences += len(items)
        rows.append([timestamp, *items])
    writer = TemporalDatabase(transactions=len(rows), items=max(1, universe))
    writer._writeDatabase(outputFile, iter(rows))
    return {"transactions": len(rows), "item_occurrences": occurrences,
            "max_item_id": universe if occurrences else None,
            "format": suffix.lstrip(".") if suffix in (".bin", ".parquet") else "tsv" if suffix == ".tsv" else "csv"}


def _convertQuantitative(source, outputFile, sep=None):
    """Preserve aligned item quantities through all supported formats."""
    from . import _quantitative_io
    rows = (_quantitative_io.read(source, sep=sep) if isinstance(source, (str, Path))
            else [_quantitative_io.normalize(row) for row in source])
    _quantitative_io.write(outputFile, rows)
    suffix = Path(outputFile).suffix.lower()
    return {"transactions": len(rows), "item_occurrences": sum(map(len, rows)),
            "max_item_id": max((item for row in rows for item in row), default=None),
            "format": suffix.lstrip('.') if suffix in ('.bin', '.parquet', '.tsv') else 'csv'}
