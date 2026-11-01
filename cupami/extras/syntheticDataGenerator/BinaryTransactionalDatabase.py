"""Generate item-presence transactions, independently of storage format.

Uses cuPAMI's uniform basket sampler in TransactionalDatabase; this is not
an implementation of IBM's association-rule workload generator.
"""
from .TransactionalDatabase import TransactionalDatabase


class BinaryTransactionalDatabase(TransactionalDatabase):
    """Binary (presence/absence) baskets; export as CSV, BIN, or Parquet.

    Arguments: transactions, items, avg_length=None, seed=None.
    Each returned row is a sorted list of unique integer item IDs.
    """
