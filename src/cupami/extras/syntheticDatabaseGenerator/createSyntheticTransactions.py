import csv
import random
from pathlib import Path
from typing import Iterator, Optional, Union


# Transactional database generator
class createSyntheticTransaction:
    def __init__(
        self,
        transactions: int,
        items: int,
        min_length: int = 1,
        max_length: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        if transactions < 0:
            raise ValueError("transactions must be non-negative")
        if items < 1:
            raise ValueError("items must be at least 1")

        if max_length is None:
            max_length = items
        if min_length < 1 or max_length < min_length or max_length > items:
            raise ValueError("transaction lengths must satisfy 1 <= min_length <= max_length <= items")

        self.transactions = transactions
        self.items = items
        self.min_length = min_length
        self.max_length = max_length
        self._random = random.Random(seed)

    def _generate_transactions(self) -> Iterator[list[int]]:
        item_ids = range(1, self.items + 1)
        for _ in range(self.transactions):
            length = self._random.randint(self.min_length, self.max_length)
            yield sorted(self._random.sample(item_ids, length))

    def createTransactionalDatabase(
        self,
        outputFile: Union[str, Path],
    ) -> None:
        output_path = Path(outputFile)
        with output_path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            for items in self._generate_transactions():
                writer.writerow(items)
