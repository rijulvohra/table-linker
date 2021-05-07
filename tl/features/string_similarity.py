import pandas as pd
import typing
import copy
import sys
import tl.features.similarity_units

from tl.exceptions import UnsupportTypeError, RequiredColumnMissingException

DEFAULT_COLUMN_COMB_NAME = ("label_clean", "kg_labels")


class StringSimilarity:
    def __init__(self, similarity_method: typing.List[str], **kwargs):
        self.similarity_units = []
        self.df = copy.deepcopy(kwargs["df"]).fillna("")
        self.target_label_column_name, self.candidate_label_column_name = kwargs.get("target_columns", (None, None))
        self.label_column = None
        for each_col in [self.target_label_column_name, self.candidate_label_column_name]:
            if each_col and each_col not in self.df:
                raise RequiredColumnMissingException("No specified column name `{}` found!".format(each_col))

        if not self.target_label_column_name:
            if "label_clean" in self.df:
                self.target_label_column_name = "label_clean"
                kwargs['target_label_column_name'] = self.target_label_column_name
                # if "label" in self.df:
                #     self.label_column = "label"
            elif "label" in self.df:
                self.target_label_column_name = "label"
                kwargs['target_label_column_name'] = self.target_label_column_name
            else:
                raise RequiredColumnMissingException("No `label` or `label_clean` column found!")
        else:
            kwargs["candidate_label_column_name"] = self.candidate_label_column_name

        if not self.candidate_label_column_name:
            if "kg_labels" in self.df:
                self.candidate_label_column_name = "kg_labels"
                kwargs['candidate_label_column_name'] = self.candidate_label_column_name
            else:
                raise RequiredColumnMissingException("No `kg_labels` column found!")
        else:
            kwargs["target_label_column_name"] = self.target_label_column_name
            # if self.target_label_column_name == 'label_clean':
            #     if "label" in self.df:
            #         self.label_column = "label"
        output_column_name = kwargs.get("output_column", None)
        if output_column_name is not None:
            self.has_output_column_name = True
            self.compared_column_names = output_column_name
        elif (self.target_label_column_name, self.candidate_label_column_name) != DEFAULT_COLUMN_COMB_NAME:
            self.has_output_column_name = False
            self.compared_column_names = self.target_label_column_name + "_" + self.candidate_label_column_name
        else:
            self.compared_column_names = None

        for each_method in similarity_method:
            # method1:a1=v1:a2=v2:a3=v3
            try:
                args = each_method.split(':')
                method_name = args[0]
                method_args = {k: v for k, v in [v.split('=') for v in args[1:]]}
                cls = getattr(tl.features.similarity_units,
                              '{}Similarity'.format(''.join([x.capitalize() for x in method_name.split('_')])))
                self.similarity_units.append(cls(tl_args=kwargs, **method_args))
            except:
                raise UnsupportTypeError("Similarity method {} does not exist or wrong arguments".format(each_method))

    @staticmethod
    def get_all_similarity_models():
        pass

    def get_similarity_score(self):
        self.df['concatenated_targets'] = list(zip(self.df[self.candidate_label_column_name],
                                                self.df[self.target_label_column_name]))
        # else:
        #     self.df['concatenated_targets'] = list(zip(self.df[self.candidate_label_column_name],
        #                                            self.df[self.target_label_column_name],
        #                                            self.df[self.label_column]))
        self.df[self.compared_column_names] = self.df['concatenated_targets'].map(lambda x: self.string_similarity(x))
        self.df.drop(columns=['concatenated_targets'], inplace=True)

        return self.df

    def string_similarity(self, pair: tuple) -> float:
        # if len(pair) == 2:
        #     og_labels = pair[0].split("|")
        #     target_labels = pair[1].split("|")
        #     is_label_used = False
        # elif len(pair) == 3:
        #     og_labels = pair[0].split('|')
        #     target_labels = pair[1].split('|')
        #     target_labels_unclean = pair[2].split('|')
        #     is_label_used = True
        og_labels = pair[0].split("|")
        target_labels = pair[1].split("|")
        max_score = 0.0

        for each_similarity_unit in self.similarity_units:
            # get max score amount all labels of candidate node and use the highest one
            for each_label in og_labels:
                for target_label in target_labels:
                    each_similarity_score = each_similarity_unit.similarity(str(target_label), str(each_label))
                    if each_similarity_score > max_score:
                        max_score = each_similarity_score
        
        # elif is_label_used:
        #     for each_similarity_unit in self.similarity_units:
        #         # get max score amount all labels of candidate node and use the highest one
        #         label_clean_score = 0.0
        #         label_unclean_score = 0.0
        #         for each_label in og_labels:
        #             for target_label in target_labels:
        #                 each_similarity_score = each_similarity_unit.similarity(str(target_label), str(each_label))
        #                 if each_similarity_score > label_clean_score:
        #                     label_clean_score = each_similarity_score
                            
        #             for target_label in target_labels_unclean:
        #                 each_similarity_score = each_similarity_unit.similarity(str(target_label), str(each_label))
        #                 if each_similarity_score > label_unclean_score:
        #                     label_unclean_score = each_similarity_score

        #         max_score = max(label_clean_score, label_unclean_score)
        return max_score
