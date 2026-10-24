from .TransactionalDatabase import TransactionalDatabase


class TemporalDatabase(TransactionalDatabase):
    """Generate timestamped baskets with optional random repeats and gaps.

    Allowed timestamp steps are 0 (repeat), 1 (next), and 2 (skip one).
    All flags default to False. Merging unions items at each repeated timestamp.
    """
    def __init__(
        self,
        transactions,
        items,
        avg_length=None,
        seed=None,
        *,
        repeatAllowed=False,
        missingAllowed=False,
        mergeTimestamps=False,
    ):
        for name, value in (
            ("repeatAllowed", repeatAllowed),
            ("missingAllowed", missingAllowed),
            ("mergeTimestamps", mergeTimestamps),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a boolean")
        if mergeTimestamps and not repeatAllowed:
            raise ValueError("mergeTimestamps=True requires repeatAllowed=True")
        super().__init__(transactions, items, avg_length, seed)
        self.repeatAllowed = repeatAllowed
        self.missingAllowed = missingAllowed
        self.mergeTimestamps = mergeTimestamps

    def _generateTransactions(self):
        timestamp = 1
        pendingTimestamp = None
        pendingItems = set()

        for index, transaction in enumerate(super()._generateTransactions()):
            if index:
                step = 1
                if self.repeatAllowed or self.missingAllowed:
                    step = self._random.randint(
                        0 if self.repeatAllowed else 1,
                        2 if self.missingAllowed else 1,
                    )
                timestamp += step

            if not self.mergeTimestamps:
                yield [timestamp, *transaction]
                continue

            if pendingTimestamp is not None and pendingTimestamp != timestamp:
                yield [pendingTimestamp, *sorted(pendingItems)]
                pendingItems.clear()

            pendingTimestamp = timestamp
            pendingItems.update(transaction)

        if pendingTimestamp is not None:
            yield [pendingTimestamp, *sorted(pendingItems)]

    def createTemporalDatabase(self, outputFile):
        self.createTransactionalDatabase(outputFile)

    def _writeDatabase(self, outputFile, transactions):
        from pathlib import Path

        from .. import _temporal_io

        suffix = Path(outputFile).suffix.lower()
        if suffix == ".bin":
            _temporal_io.write_binary(outputFile, transactions, self.items)
        elif suffix == ".parquet":
            _temporal_io.write_parquet(outputFile, transactions, self.items)
        else:
            super()._writeDatabase(outputFile, transactions)
