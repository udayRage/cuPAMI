from pathlib import Path
import sys

if __name__ == '__main__' and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = 'cupami.fuzzyFrequentPattern'

from ._multi_api import _MultiMiner, _main


class cuGPUMulti(_MultiMiner):
    _grouped = True


if __name__ == '__main__':
    _main(cuGPUMulti)
