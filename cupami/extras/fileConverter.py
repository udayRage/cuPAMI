"""Convert database files through one interface selected by data type."""

from pathlib import Path


def convertFile(source, outputFile, dataType="transactional", sep=None):
    """Select database semantics by dataType and storage format by extension.

    Currently supported data types: transactional. Each future type adds a
    handler here, keeping the same public interface for callers.
    """
    handlers = {"transactional": _convertTransactional}
    if not isinstance(dataType, str) or dataType.lower() not in handlers:
        raise ValueError(f"Unsupported data type {dataType!r}; supported: transactional")
    return handlers[dataType.lower()](source, outputFile, sep=sep)


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
    if suffix not in (".csv", ".bin", ".parquet", ".txt", ".dat", ""):
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
            "format": suffix.lstrip(".") if suffix in (".bin", ".parquet") else "csv"}
