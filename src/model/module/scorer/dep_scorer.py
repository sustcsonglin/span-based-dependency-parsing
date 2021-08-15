import torch.nn as nn
from .module.biaffine import BiaffineScorer
from .module.triaffine import TriaffineScorer
import torch
import logging
log = logging.getLogger(__name__)



class DepScorer(nn.Module):
    def __init__(self, conf, fields, input_dim):
        super(DepScorer, self).__init__()
        self.conf = conf


        self.rel_scorer = BiaffineScorer(n_in=input_dim, n_out=conf.n_mlp_rel, bias_x=True, bias_y=True,
                                             dropout=conf.mlp_dropout, n_out_label=fields.get_vocab_size("rel"),
                                             scaling=conf.scaling)


        if self.conf.use_arc:
            self.arc_scorer = BiaffineScorer(n_in=input_dim, n_out=conf.n_mlp_arc, bias_x=True, bias_y=False, dropout=conf.mlp_dropout, scaling=conf.scaling)
            log.info("Use arc score")
            if self.conf.use_sib:
                log.info("Use sib score")
                self.sib_scorer = TriaffineScorer(n_in=input_dim, n_out=conf.n_mlp_sib, bias_x=True, bias_y=True,
                                                  dropout=conf.mlp_dropout)

        if self.conf.use_span:
            log.info('use span score')
            if self.conf.span_scorer_type == 'biaffine':
                assert not self.conf.use_sib
                self.span_scorer = BiaffineScorer(n_in=input_dim, n_out=conf.n_mlp_arc, bias_x=True, bias_y=True, dropout=conf.mlp_dropout, scaling=conf.scaling)

            elif self.conf.span_scorer_type == 'triaffine':
                assert not self.conf.use_sib
                self.span_scorer = TriaffineScorer(n_in=input_dim, n_out=conf.n_mlp_arc, bias_x=True, bias_y=True, dropout=conf.mlp_dropout, )

            elif self.conf.span_scorer_type == 'headsplit':
                assert self.conf.use_arc
                self.span_scorer_left = BiaffineScorer(n_in=input_dim, n_out=conf.n_mlp_arc, bias_x=True, bias_y=True, dropout=conf.mlp_dropout, scaling=conf.scaling)
                self.span_scorer_right = BiaffineScorer(n_in=input_dim, n_out=conf.n_mlp_arc, bias_x=True, bias_y=True, dropout=conf.mlp_dropout, scaling=conf.scaling)
            else:
                raise NotImplementedError



    def forward(self, ctx):
        x = ctx['encoded_emb']
        x_f, x_b = x.chunk(2, -1)
        x_boundary = torch.cat((x_f[:, :-1], x_b[:, 1:]), -1)

        if self.conf.use_arc:
            ctx['s_arc'] = self.arc_scorer(x[:, :-1])
            if self.conf.use_sib:
                ctx['s_sib'] = self.sib_scorer(x[:, :-1])

        if self.conf.use_span:
            if self.conf.span_scorer_type == 'headsplit':
                ctx['s_bd_left'] = self.span_scorer_left.forward_v2(x[:,1:], x_boundary)
                ctx['s_bd_right'] = self.span_scorer_right.forward_v2(x[:,1:], x_boundary)

            elif self.conf.span_scorer_type == 'biaffine':
                # LSTM minus features
                span_repr = (x_boundary.unsqueeze(1) - x_boundary.unsqueeze(2))
                batch, seq_len = span_repr.shape[:2]
                span_repr2 = span_repr.reshape(batch, seq_len * seq_len, -1)
                ctx['s_span_head_word'] = self.span_scorer.forward_v2(span_repr2, x[:, 1:-1]).reshape(batch, seq_len,
                                                                                                  seq_len, -1)
            elif self.conf.span_scorer_type == 'triaffine':
                ctx['s_span_head_word'] = self.span_scorer.forward2(x[:, 1:-1], x_boundary)
            else:
                raise NotImplementedError

        ctx['s_rel'] = self.rel_scorer(x[:, :-1])




