"""Convert binary, temporal, and quantitative datasets without CUDA.

Examples:
    python -m cupami.extras.datasetConverter baskets.csv baskets.bin
    python -m cupami.extras.datasetConverter baskets.bin baskets.parquet
    python -m cupami.extras.datasetConverter baskets.parquet baskets.tsv
    python -m cupami.extras.datasetConverter quantities.csv quantities.bin --data-type quantitative

Uses cuPAMI's versioned BIN schemas and Apache Parquet list columns:
https://arrow.apache.org/docs/python/parquet.html
"""
import argparse
import json

from .fileConverter import convertFile, convertTransactions

__all__ = ['convertFile', 'convertTransactions']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('source', help='Input CSV, TSV, text, BIN, or Parquet file')
    parser.add_argument('output', help='Output file; extension selects its format')
    parser.add_argument('--data-type', choices=['binary', 'transactional', 'temporal', 'quantitative'], default='binary')
    parser.add_argument('--sep', help=r'Input text delimiter, e.g. "," or "\t"; default: detect')
    parser.add_argument('--output-sep', help=r'Output text delimiter; default: comma (tab for .tsv)')
    args = parser.parse_args(argv)
    def delimiter(value):
        return '\t' if value == r'\t' else value
    try:
        summary = convertFile(args.source, args.output, dataType=args.data_type,
                              sep=delimiter(args.sep), outputSep=delimiter(args.output_sep))
    except (ValueError, OSError, ImportError) as error:
        parser.exit(2, f'Conversion failed: {error}\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
