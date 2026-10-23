from .TransactionalDatabase import TransactionalDatabase


class TemporalDatabase(TransactionalDatabase):
    def __init__(
        self,
        transactions,
        items,
        avg_length=None,
        seed=None,
        *,
        timestampStep=1,
        repeatChance=0.0,
        mergeTimestamps=False,
    ):
        if isinstance(timestampStep, bool) or not isinstance(timestampStep, int):
            raise TypeError("timestampStep must be an integer")
        if timestampStep < 1:
            raise ValueError("timestampStep must be at least 1")
        if isinstance(repeatChance, bool) or not isinstance(
            repeatChance, (int, float)
        ):
            raise TypeError("repeatChance must be a number")
        if not 0 <= repeatChance <= 1:
            raise ValueError("repeatChance must be between 0 and 1")
        if not isinstance(mergeTimestamps, bool):
            raise TypeError("mergeTimestamps must be a boolean")

        super().__init__(transactions, items, avg_length, seed)
        self.timestampStep = timestampStep
        self.repeatChance = repeatChance
        self.mergeTimestamps = mergeTimestamps

    def _generateTransactions(self):
        timestamp = 1
        pendingTimestamp = None
        pendingItems = set()

        for index, transaction in enumerate(super()._generateTransactions()):
            if index and not (
                self.repeatChance
                and self._random.random() < self.repeatChance
            ):
                timestamp += self.timestampStep

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
