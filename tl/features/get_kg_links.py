import pandas as pd
from tl.exceptions import RequiredInputParameterMissingException
from tl.file_formats_validator import FFV
import sys
import numpy as np


def get_kg_links(score_column, file_path=None, df=None, label_column='label', top_k=5, k_rows=False):
    if file_path is None and df is None:
        raise RequiredInputParameterMissingException(
            'One of the input parameters is required: {} or {}'.format("file_path", "df"))

    if score_column is None:
        raise RequiredInputParameterMissingException(
            'One of the input parameters is required: {}'.format('score_column'))

    if file_path:
        df = pd.read_csv(file_path, dtype=object)
    df.fillna("", inplace=True)
    df = df.astype(dtype={score_column: "float64"})
    ffv = FFV()
    if not (ffv.is_candidates_file(df)):
        raise UnsupportTypeError("The input file is not a candidate file!")

    topk_df = df.groupby(['column', 'row']).apply(lambda x: x.sort_values([score_column], ascending=False)) \
        .reset_index(drop=True)

    final_list = []
    grouped_obj = topk_df.groupby(['row', 'column'])
    for cell in grouped_obj:
        cell[1].drop_duplicates(subset='kg_id', inplace=True)
        if not(k_rows):
            _ = {}
            _['column'] = cell[0][1]
            _['row'] = cell[0][0]
            _['label'] = cell[1][label_column].unique()[0]
            _['kg_id'] = '|'.join(list(cell[1]['kg_id'])[:top_k])
            _['kg_label'] = '|'.join(list(cell[1]['kg_labels'])[:top_k])
            _['kg_description'] = '|'.join(list(cell[1]['kg_descriptions'])[:top_k])
            _['ranking_score'] = '|'.join([str(round(score, 2)) for score in list(cell[1][score_column])[:top_k]])
            final_list.append(_)
        else:
            reset_index_df = cell[1].reset_index(drop=True)
            topk_df_row = reset_index_df.head(top_k)
            topk_df_row = topk_df_row.astype(dtype={'evaluation_label': "int32"})
            #print(cell[1]['evaluation_label'], file=sys.stderr)
            # cell[1] = cell[1].astype(dtype={'evaluation_label': "int32"})
            correct_gt = reset_index_df[reset_index_df['evaluation_label']=='1']
            if (topk_df_row.iloc[0, 14] == 1):
                topk_df_row['correct'] = [1 for _ in range(topk_df_row.shape[0])]
                topk_df_row['rank'] = [i+1 for i in range(topk_df_row.shape[0])]
            elif (1 in topk_df_row['evaluation_label'].tolist()):
                topk_df_row['correct'] = [0 for _ in range(topk_df_row.shape[0])]
                topk_df_row['rank'] = [i+1 for i in range(topk_df_row.shape[0])]
            else:
                topk_df_row['correct'] = [-1 for _ in range(topk_df_row.shape[0])]
                topk_df_row['rank'] = [i+1 for i in range(topk_df_row.shape[0])]
                correct_gt = reset_index_df[reset_index_df['evaluation_label']=='1']
                if correct_gt.shape[0] >= 1:
                    correct_gt['correct'] = -1
                    correct_gt['rank'] = correct_gt.index[0]
                    topk_df_row = pd.concat([topk_df_row, correct_gt])
                
            final_list.extend(topk_df_row.to_dict(orient='records'))

    odf = pd.DataFrame(final_list)
    return odf
