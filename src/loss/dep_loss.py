from ..inside import *
from supar.utils.common import *
from .get_score import *
import torch
import torch.nn.functional as F

class DepLoss():
    def __init__(self, conf):
        self.conf = conf
        self.inside = eisner


    @classmethod
    def label_loss(self, ctx, reduction='mean'):
        s_rel = ctx['s_rel']
        gold_rel = ctx['rel']
        gold_arc = ctx['head']
        if len(s_rel.shape) == 4:
            return F.cross_entropy(s_rel[gold_arc[:, 0], gold_arc[:, 1], gold_arc[:, 2]] , torch.as_tensor(gold_rel[:, -1], device=s_rel.device, dtype=torch.long),reduction=reduction)
        elif len(s_rel.shape) == 3:
            return F.cross_entropy(s_rel[gold_arc[:, 0], gold_arc[:, 1]], torch.as_tensor(gold_rel[:, -1], device=s_rel.device, dtype=torch.long),reduction=reduction)
        else:
            raise AssertionError

    @classmethod
    def get_pred_rels(self,ctx):
        arc_preds = ctx['arc_pred']
        s_rel = ctx['s_rel']
        ctx['rel_pred'] = s_rel.argmax(-1).gather(-1, arc_preds.unsqueeze(-1)).squeeze(-1)

    # for evaluation.
    @classmethod
    def _transform(self, ctx):
        arc_preds, arc_golds = ctx['arc_pred'], ctx['head']
        rel_preds, rel_golds = ctx['rel_pred'], ctx['rel']

        arc_golds = torch.as_tensor(arc_golds, device=arc_preds.device, dtype=arc_preds.dtype)
        rel_golds = torch.as_tensor(rel_golds, device=rel_preds.device, dtype=rel_preds.dtype)

        arc_gold = arc_preds.new_zeros(*arc_preds.shape).fill_(-1)
        arc_gold[arc_golds[:, 0], arc_golds[:, 1]] = arc_golds[:, 2]

        rel_gold = rel_preds.new_zeros(*rel_preds.shape).fill_(-1)
        rel_gold[rel_golds[:, 0], rel_golds[:, 1]] = rel_golds[:, 2]
        mask_dep = arc_gold.ne(-1)

        #ignore punct.
        if 'is_punct' in ctx:
            mask_punct = ctx['is_punct'].nonzero()
            mask_dep[mask_punct[:, 0], mask_punct[:, 1] + 1] = False
        ctx['arc_gold'] = arc_gold
        ctx['rel_gold'] = rel_gold
        ctx['mask_dep'] = mask_dep

    def max_margin_loss(self, ctx):
        with torch.no_grad():
            self.inside(ctx, max_margin=True)
        gold_score = u_score(ctx)
        predict_score = predict_score_mm(ctx)
        return (predict_score - gold_score)/ctx['seq_len'].sum()

    def crf_loss(self, ctx):
        logZ = self.inside(ctx).sum()
        gold_score = u_score(ctx)
        return  (logZ - gold_score)/ ctx['seq_len'].sum()

    def local_loss(self, ctx):
        raise NotImplementedError

    def loss(self, ctx):
        if self.conf.loss_type == 'mm':
            tree_loss = self.max_margin_loss(ctx)
        elif self.conf.loss_type == 'crf':
            tree_loss = self.crf_loss(ctx)
        elif self.conf.loss_type == 'local':
            tree_loss = self.local_loss(ctx)
        label_loss = self.label_loss(ctx)
        return tree_loss + label_loss

    def decode(self, ctx):
        self.inside(ctx, decode=True)
        self.get_pred_rels(ctx)
        self._transform(ctx)


class Dep1OSpan(DepLoss):
    def __init__(self, conf):
        super(Dep1OSpan, self).__init__(conf)
        self.inside = es4dep

class Dep2O(DepLoss):
    def __init__(self, conf):
        super(Dep2O, self).__init__(conf)
        self.inside = eisner2o

class Dep2OHeadSplit(DepLoss):
    def __init__(self, conf):
        super(Dep2OHeadSplit, self).__init__(conf)
        self.inside = eisner2o_headsplit

class Dep1OHeadSplit(DepLoss):
    def __init__(self, conf):
        super(Dep1OHeadSplit, self).__init__(conf)
        self.inside = eisner_headsplit

class Span(DepLoss):
    def __init__(self, conf):
        super(Span, self).__init__(conf)
        self.inside = span_inside


