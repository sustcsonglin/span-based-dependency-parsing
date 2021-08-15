from supar.utils.alg import kmeans
from supar.utils.data import Sampler
from fastNLP.core.field import Padder
import numpy as np




def get_sampler(lengths, max_tokens, n_buckets, shuffle=True, distributed=False, evaluate=False):
    buckets = dict(zip(*kmeans(lengths, n_buckets)))
    return Sampler(buckets=buckets,
                   batch_size=max_tokens,
                   shuffle=shuffle,
                   distributed=distributed,
                   evaluate=evaluate)



class SiblingPadder(Padder):
    def __call__(self, contents, field_name, field_ele_dtype, dim: int):
        # max_sent_length = max(rule.shape[0] for rule in contents)
        batch_size = sum(len(r) for r in contents)
        padded_array = np.full((batch_size, 4), fill_value=self.pad_val,
                               dtype=np.long)

        i = 0
        for b_idx, relations in enumerate(contents):
            for (head, child, sibling, _)  in relations:
                padded_array[i] = np.array([b_idx, head, child, sibling])
                i+=1

        return padded_array




class ConstAsDepPadder(Padder):
    def __call__(self, contents, field_name, field_ele_dtype, dim: int):
        # max_sent_length = max(rule.shape[0] for rule in contents)
        batch_size = sum(len(r) for r in contents)
        padded_array = np.full((batch_size, 4), fill_value=self.pad_val,
                               dtype=np.long)
        i = 0
        for b_idx, relations in enumerate(contents):
            for (head, child, sibling, *_)  in relations:
                padded_array[i] = np.array([b_idx, head, child, sibling])
                i+=1
        return padded_array

class GrandPadder(Padder):
    def __call__(self, contents, field_name, field_ele_dtype, dim: int):
        batch_size = sum(len(r) for r in contents)
        padded_array = np.full((batch_size, 4), fill_value=self.pad_val,
                               dtype=np.float)
        i = 0
        for b_idx, relations in enumerate(contents):
            for (head, child, _, grandparent, *_) in relations:
                padded_array[i] = np.array([b_idx, head, child, grandparent])
                i += 1

        return padded_array

class SpanPadderCAD(Padder):
    def __call__(self, contents, field_name, field_ele_dtype, dim: int):
        batch_size = sum((len(r) * 2 - 1) for r in contents)
        padded_array = np.full((batch_size, 6), fill_value=self.pad_val,
                               dtype=np.float)
        i = 0

        ## 0 stands for inherent
        ## 1 stands for noninherent
        for b_idx, span in enumerate(contents):
            for (head, child, ih_start, ih_end, ni_start, ni_end) in span:
                if not ih_start == -1:
                    padded_array[i] = np.array([b_idx, head, child, ih_start, ih_end, 0])
                    i += 1
                padded_array[i] = np.array([b_idx, head, child, ni_start, ni_end, 1])
                i += 1

        assert i == batch_size
        return padded_array




class SpanHeadPadder(Padder):
    def __call__(self, contents, field_name, field_ele_dtype, dim: int):
        batch_size = sum(len(r) for r in contents)
        padded_array = np.full((batch_size, 4), fill_value=self.pad_val,
                               dtype=np.float)
        i = 0
        for b_idx, relations in enumerate(contents):
            for (left, right, head) in relations:
                padded_array[i] = np.array([b_idx, left, right, head])
                i += 1
        return padded_array




