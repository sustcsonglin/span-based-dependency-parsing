from fastNLP.core.field import Padder
import numpy as np


def set_padder(datasets, name, padder):
    for _, dataset in datasets.items():
        dataset.add_field(name, dataset[name].content, padder=padder, ignore_type=True)


class DepSibPadder(Padder):
    def __call__(self, contents, field_name, field_ele_dtype, dim: int):
        # max_sent_length = max(rule.shape[0] for rule in contents)
        padded_array = []
        # dependency head or relations.
        for b_idx, dep in enumerate(contents):
            for (child_idx, (head_idx, sib_ix)) in enumerate(dep):
                # -1 means no sib;
                if sib_ix != -1:
                    padded_array.append([b_idx, child_idx + 1, head_idx, sib_ix])
                else:
                    pass
        return np.array(padded_array)





class DepPadder(Padder):
    def __call__(self, contents, field_name, field_ele_dtype, dim: int):
            # max_sent_length = max(rule.shape[0] for rule in contents)
            padded_array = []
            # dependency head or relations.
            for b_idx, dep in enumerate(contents):
                for (child_idx, dep_idx) in enumerate(dep):
                    padded_array.append([b_idx, child_idx + 1, dep_idx])
            return np.array(padded_array)




class SpanHeadWordPadder(Padder):
    def __call__(self, contents, field_name, field_ele_dtype, dim: int):
        padded_array = []
        for b_idx, relations in enumerate(contents):
            for (left, right, _, head) in relations:
                padded_array.append([b_idx, left, right, head])
        return np.array(padded_array)





