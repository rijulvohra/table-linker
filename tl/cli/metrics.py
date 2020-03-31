import sys
import argparse


def parser():
    return {
        'help': 'computes the precision, recall and f1 score for the tl pipeline'
    }


def add_arguments(parser):
    """
    Parse Arguments
    Args:
        parser: (argparse.ArgumentParser)

    """
    parser.add_argument('-c', '--column', action='store', type=str, dest='column', required=True,
                        help='column name with ranking scores')

    parser.add_argument('-k', action='store', type=int, dest='k', default=1,
                        help='calculate recall at top k, default is 1')

    parser.add_argument('input_file', nargs='?', type=argparse.FileType('r'), default=sys.stdin)


def run(**kwargs):
    from tl.evaluation import evaluation
    import pandas as pd

    df = pd.read_csv(kwargs['input_file'], dtype=object)

    odf = evaluation.metrics(kwargs['column'], k=kwargs['k'], df=df)
    odf.to_csv(sys.stdout, index=False)