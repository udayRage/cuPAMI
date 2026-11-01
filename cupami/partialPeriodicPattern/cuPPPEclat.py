"""Depth-first CUDA partial-periodic pattern mining.
GPU reference: PAMI gPPMiner, bitset intersections and temporal scans.
https://github.com/UdayLab/PAMI/blob/48d0c25fb5f1e4ff32e797fa00873033921f1aa7/PAMI/partialPeriodicPattern/cuda/gPPMiner.py
CPU model/API: PAMI PPP_ECLAT; depth-first prefix extension.
https://github.com/UdayLab/PAMI/blob/48d0c25fb5f1e4ff32e797fa00873033921f1aa7/PAMI/partialPeriodicPattern/basic/PPP_ECLAT.py
Local variant uses row bitsets plus timestamps and counts only consecutive
occurrence gaps, excluding boundary gaps. See docs/algorithm-references.md.
"""
from ._partial import PartialInput
from ..periodicFrequentPattern.cuPFEclat import cuPFEclat


class cuPPPEclat(PartialInput):
    """Mine with (iFile, minPS, period, sep=None); return periodic supports."""
    _BATCH_SIZE = 4096
    mine = cuPFEclat.mine
