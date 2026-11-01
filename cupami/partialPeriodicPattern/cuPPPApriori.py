"""Level-wise CUDA partial-periodic pattern mining.
GPU reference: PAMI gPPMiner's bitset intersection/periodic-gap scan.
https://github.com/UdayLab/PAMI/blob/48d0c25fb5f1e4ff32e797fa00873033921f1aa7/PAMI/partialPeriodicPattern/cuda/gPPMiner.py
Local breadth-first variant follows PPP_ECLAT's consecutive-occurrence
support definition, excluding boundary gaps; gPPMiner's kernel includes
a start gap. See _partial.py and docs/algorithm-references.md.
"""
from ._partial import PartialInput
from ..periodicFrequentPattern.cuPFApriori import cuPFApriori


class cuPPPApriori(PartialInput):
    """Mine with (iFile, minPS, period, sep=None); return periodic supports."""
    _BATCH_SIZE = 4096
    mine = cuPFApriori.mine
