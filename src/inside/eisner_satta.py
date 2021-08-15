from .fn import *
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from ..loss.get_score import augment_score



# O(n4) span+arc
@torch.enable_grad()
def es4dep(ctx, decode=False, max_margin=False):
    if max_margin:
        augment_score(ctx)

    lens = ctx['seq_len']

    dependency = ctx['s_arc']


    B, seq_len = dependency.shape[:2]
    head_score = ctx['s_span_head_word']

    if decode:
        dependency = dependency.detach().clone().requires_grad_(True)

    if max_margin:
        dependency = dependency.detach().clone().requires_grad_(True)
        head_score = head_score.detach().clone().requires_grad_(True)

    dep = dependency[:, 1:, 1:].contiguous()
    root = dependency[:, 1:, 0].contiguous()

    viterbi = decode or max_margin

    N = seq_len
    H = N - 1

    s = dependency.new_zeros(B, N, N, H).fill_(-1e9)
    s_close = dependency.new_zeros(B, N, N, H).fill_(-1e9)
    s_need_dad = dependency.new_zeros(B, N, N, H).fill_(-1e9)

    s_close[:, torch.arange(N - 1), torch.arange(N - 1) + 1, torch.arange(N - 1)] = head_score[:, torch.arange(N - 1), torch.arange(N - 1) + 1, torch.arange(N - 1)]

    s[:, torch.arange(N - 1), torch.arange(N - 1) + 1, torch.arange(N - 1)] = 0

    s_need_dad[:, torch.arange(N - 1), torch.arange(N - 1) + 1, :] = dep[:, torch.arange(N - 1)] + s_close[:, torch.arange(N - 1), torch.arange(N - 1) + 1, torch.arange(N - 1)].unsqueeze(-1)

    def merge(left, right, left_need_dad, right_need_dad):
        left = (left + right_need_dad)
        right = (right + left_need_dad)
        if viterbi:
            headed = torch.stack([left.max(2)[0], right.max(2)[0]])
            return headed.max(0)[0]
        else:
            headed = torch.stack([left.logsumexp(2), right.logsumexp(2)])
            return headed.logsumexp(0)

    def seek_head(a, b):
        if viterbi:
            tmp = (a + b).max(-2)[0]
        else:
            tmp = (a + b).logsumexp(-2)
        return tmp


    for w in range(2, N):
        n = N - w
        left = stripe_version2(s, n, w - 1, (0, 1))
        right = stripe_version2(s, n, w - 1, (1, w), 0)
        left_need_dad = stripe_version2(s_need_dad, n, w - 1, (0, 1))
        right_need_dad = stripe_version2(s_need_dad, n, w - 1, (1, w), 0)
        headed = checkpoint(merge, left.clone(), right.clone(), left_need_dad.clone(),right_need_dad.clone())
        diagonal_copy_v2(s, headed, w)
        headed = headed + diagonal_v2(head_score, w)
        diagonal_copy_v2(s_close, headed, w)

        if w < N - 1:
            u = checkpoint(seek_head, headed.unsqueeze(-1), stripe_version5(dep, n, w))
            diagonal_copy_(s_need_dad, u, w)

    logZ = (s_close[torch.arange(B), 0, lens] + root)
    if viterbi:
        logZ = logZ.max(-1)[0]
    else:
        logZ = logZ.logsumexp(-1)

    #crf loss
    if not decode and not max_margin:
        return logZ

    logZ.sum().backward()

    if decode:
        predicted_arc = s.new_zeros(B, seq_len).long()
        arc = dependency.grad.nonzero()
        predicted_arc[arc[:, 0], arc[:, 1]] = arc[:, 2]
        ctx['arc_pred'] = predicted_arc

    if max_margin:
        ctx['s_arc_grad'] = dependency.grad
        if head_score is not None:
            ctx['s_span_head_word_grad'] = head_score.grad
