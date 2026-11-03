"""Shared base-item traversal for multiple fuzzy frequent pattern mining.

Accompanies "cuMFFP-Miner: GPU-Accelerated Multiple Fuzzy Frequent Pattern
Mining via Shared Base-Item Traversal" by Mithun Thangaraj and Rage Uday Kiran,
The University of Aizu, Japan.
"""
from pathlib import Path
import sys

if __name__ == '__main__' and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = 'cupami.fuzzyFrequentPattern'

from ._multi_api import _MultiMiner, _main


class cuMFFIMiner(_MultiMiner):
    _grouped = True


if __name__ == '__main__':
    _main(cuMFFIMiner)
