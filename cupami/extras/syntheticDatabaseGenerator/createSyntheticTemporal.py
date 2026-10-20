from .createSyntheticTransactional import syntheticTransactional


class syntheticTemporal(syntheticTransactional):
    def __init__(
        self,
        transactions,
        items,
        min_length=1,
        max_length=None,
        seed=None,
        *,
        timestampStep=1,
        duplicateTimestampProbability=0.0,
        mergeDuplicateTimestamps=False,
    ):
        if isinstance(timestampStep, bool) or not isinstance(timestampStep, int):
            raise TypeError("timestampStep must be an integer")
        if timestampStep < 1:
            raise ValueError("timestampStep must be at least 1")
        if isinstance(duplicateTimestampProbability, bool) or not isinstance(
            duplicateTimestampProbability, (int, float)
        ):
            raise TypeError("duplicateTimestampProbability must be a number")
        if not 0 <= duplicateTimestampProbability <= 1:
            raise ValueError("duplicateTimestampProbability must be between 0 and 1")
        if not isinstance(mergeDuplicateTimestamps, bool):
            raise TypeError("mergeDuplicateTimestamps must be a boolean")

        super().__init__(transactions, items, min_length, max_length, seed)
        self.timestampStep = timestampStep
        self.duplicateTimestampProbability = duplicateTimestampProbability
        self.mergeDuplicateTimestamps = mergeDuplicateTimestamps

    def _generateTransactions(self):
        timestamp = 1
        pendingTimestamp = None
        pendingItems = set()

        for index, transaction in enumerate(super()._generateTransactions()):
            if index and not (
                self.duplicateTimestampProbability
                and self._random.random() < self.duplicateTimestampProbability
            ):
                timestamp += self.timestampStep

            if not self.mergeDuplicateTimestamps:
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

        if Path(outputFile).suffix.lower() in (".bin", ".parquet"):
            raise ValueError("BIN and Parquet export currently support transactional data only")
        super()._writeDatabase(outputFile, transactions)


# Keep existing imports working.
createSyntheticTemporal = syntheticTemporal
