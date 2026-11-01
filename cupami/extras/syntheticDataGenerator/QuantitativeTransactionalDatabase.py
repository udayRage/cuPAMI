"""Generate item/quantity baskets with cuPAMI's uniform basket sampler.

Quantities are independent uniform integer draws, not utility values or
fuzzy memberships. File schemas are defined in extras._quantitative_io.
"""
from numbers import Integral

from .TransactionalDatabase import TransactionalDatabase
from .. import _quantitative_io


class QuantitativeTransactionalDatabase(TransactionalDatabase):
    """Return rows of {item_id: quantity}, with unique items per basket.

    min_quantity and max_quantity are inclusive positive integer bounds.
    CSV uses aligned items:quantities lists; BIN and Parquet preserve both.
    """
    def __init__(self, transactions, items, avg_length=None, seed=None, *,
                 min_quantity=1, max_quantity=10):
        for name, value in (('min_quantity', min_quantity), ('max_quantity', max_quantity)):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f'{name} must be an integer')
            if not 1 <= value <= 2**53:
                raise ValueError(f'{name} must be in 1..2**53')
        if min_quantity > max_quantity:
            raise ValueError('min_quantity must not exceed max_quantity')
        super().__init__(transactions, items, avg_length, seed)
        self.min_quantity = int(min_quantity)
        self.max_quantity = int(max_quantity)

    def _generateTransactions(self):
        for basket in super()._generateTransactions():
            yield {item: self._random.randint(self.min_quantity, self.max_quantity)
                   for item in basket}

    def createQuantitativeDatabase(self, outputFile):
        """Generate directly to a file without retaining all transactions."""
        self.createTransactionalDatabase(outputFile)

    def _writeDatabase(self, outputFile, transactions):
        _quantitative_io.write(outputFile, transactions)
