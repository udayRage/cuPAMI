"""cuDF load/split/explode pipeline adapted from udayRage/cuda_pami.

Source: src/algorithms/frequent/fpminer-bitset.py, commit 2ed352a,
GPL-3.0, https://github.com/udayRage/cuda_pami.
cuPAMI retains its list-column Parquet format and transaction deduplication.
"""

import re


class UnsupportedTextLayout(ValueError):
    """Use the general CSV reader for quoted or nonnumeric item labels."""


def _pairs(frame):
    import cupy as cp
    import cudf

    frame = frame[frame["items"].list.len() > 0].reset_index(drop=True)
    count = len(frame)
    frame["_row"] = cp.arange(count, dtype=cp.uint64)
    pairs = frame[["_row", "items"]].explode("items").rename(columns={"items": "item"})
    pairs["item"] = pairs["item"].astype("str").str.strip()
    pairs = pairs.drop_duplicates(subset=["_row", "item"])
    items = sorted(pairs["item"].unique().to_arrow().to_pylist())
    mapping = cudf.DataFrame({"item": cudf.Series(items, dtype="str"),
                              "_column": cp.arange(len(items), dtype=cp.uint32)})
    pairs = pairs.merge(mapping, on="item")
    return items, pairs["_row"].values, pairs["_column"].values, count


def read_text(path, separator):
    import cudf

    frame = cudf.read_csv(
        str(path), sep="\x1f", header=None, names=["items"], dtype=["str"],
        quoting=3, keep_default_na=False, na_filter=False,
    )
    frame["items"] = frame["items"].str.strip()
    frame = frame[frame["items"].str.len() > 0]
    # The generator emits unquoted integer baskets. Preserve the general CPU
    # parser for other CSV layouts rather than changing their interpretation.
    # re.escape emits a backslash followed by a literal tab, which cuDF's
    # regex compiler rejects. Use its explicit tab escape instead.
    delimiter = r"\t" if separator == "\t" else re.escape(separator)
    pattern = rf"^[0-9]+(?:\s*{delimiter}\s*[0-9]+)*$"
    if not bool(frame["items"].str.contains(pattern, regex=True).all()):
        raise UnsupportedTextLayout("GPU text reader accepts unquoted integer baskets")
    frame["items"] = frame["items"].str.split(separator)
    return _pairs(frame)


def read_parquet(path):
    import cudf
    import numpy as np

    frame = cudf.read_parquet(str(path))
    if "items" not in frame.columns:
        raise ValueError("Invalid Parquet file: missing items column")
    dtype = frame["items"].dtype
    if not isinstance(dtype, cudf.ListDtype) or dtype.element_type != np.dtype("uint32"):
        raise ValueError("Invalid Parquet file: items must be a list of uint32 IDs")
    if bool(frame["items"].isna().any()) or bool(frame["items"].list.leaves.isna().any()):
        raise ValueError("Invalid Parquet file: null transactions or item IDs")
    return _pairs(frame)
