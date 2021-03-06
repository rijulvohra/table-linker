import sys
import argparse
import traceback
import tl.exceptions


def parser():
    return {
        'help': """Compute tf-idf score based on the candidate nodes' edges similarity."""
    }


def add_arguments(parser):
    """
    Parse Arguments
    Args:
        parser: (argparse.ArgumentParser)

    """

    parser.add_argument('input_file', nargs='?', type=argparse.FileType('r'), default=sys.stdin)
    parser.add_argument('-o', '--output-column', action='store', type=str, dest='output_column_name', default="tf_idf_score",
                        help='the output column name where the normalized scores will be stored.Default is tf_idf_score')
    parser.add_argument('--similarity-column', action='store', nargs='?', dest='similarity_column', required=False,
                        default="retrieval_score_normalized",
                        help="The similarity column used to support as weight of computing tf-idf scores. "
                             "If not specified, default will use `retrieval_score_normalized`")

    # TODO: add support to use different high precision candidates method
    # parser.add_argument('--high-precision-candidates-method', action='store', nargs='?',
    #                     dest='high_precision_candidates_method',
    #                     required=False, default="from_exact_match",
    #                     help="The method to choose the high precision candidates)


def run(**kwargs):
    from tl.features import tfidf
    try:
        tfidf_unit = tfidf.TFIDF(**kwargs)
        odf = tfidf_unit.compute_tfidf()
        odf.to_csv(sys.stdout, index=False)
    except:
        message = 'Command: compute-tf-idf\n'
        message += 'Error Message:  {}\n'.format(traceback.format_exc())
        raise tl.exceptions.TLException(message)
