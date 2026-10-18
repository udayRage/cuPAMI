import csv
import random
from pathlib import Path


class createSyntheticTransaction:
    def __init__(self, transactions, items, min_length=1, max_length=None, seed=None):
        if transactions < 0:
            raise ValueError("transactions must be non-negative")
        if items < 1:
            raise ValueError("items must be at least 1")

        if max_length is None:
            max_length = items
        if min_length < 1 or max_length < min_length or max_length > items:
            raise ValueError(
                "transaction lengths must satisfy 1 <= min_length <= max_length <= items"
            )

        self.transactions = transactions
        self.items = items
        self.min_length = min_length
        self.max_length = max_length
        self._random = random.Random(seed)
        self._database = None

    def create(self):
        self._database = list(self._generateTransactions())

    def getTransactions(self):
        if self._database is None:
            raise RuntimeError("Call create() before getTransactions()")
        return [transaction.copy() for transaction in self._database]

    def save(self, outputFile):
        if self._database is None:
            raise RuntimeError("Call create() before save()")
        with Path(outputFile).open("w", newline="", encoding="utf-8") as output:
            csv.writer(output).writerows(self._database)

    def _generateTransactions(self):
        itemIds = range(1, self.items + 1)
        for _ in range(self.transactions):
            length = self._random.randint(self.min_length, self.max_length)
            yield sorted(self._random.sample(itemIds, length))

    def createTransactionalDatabase(self, outputFile):
        with Path(outputFile).open("w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            for transaction in self._generateTransactions():
                writer.writerow(transaction)
