import torch
from supar.utils.fn import *
from src.loss.get_score import augment_score

#https://github.com/yzhangcs/parser/blob/main/supar/utils/alg.py
#I made some minor change.
@torch.enable_grad()
def eisner(ctx, decode=False, max_margin=False):
    if max_margin:
        augment_score(ctx)

    scores = ctx['s_arc']
    lens = ctx['seq_len']
    if decode or max_margin:
        scores_origin = scores.detach().clone().requires_grad_(True)
    else:
        scores_origin = scores

    # the end position of each sentence in a batch
    batch_size, seq_len, _ = scores.shape
    # [seq_len, seq_len, batch_size]
    scores = scores_origin.permute(2, 1, 0)
    s_i = torch.full_like(scores, float('-inf'))
    s_c = torch.full_like(scores, float('-inf'))
    s_c.diagonal().fill_(0)
    # set the scores of arcs excluded by cands to -inf
    viterbi = decode or max_margin

    for w in range(1, seq_len):
        # n denotes the number of spans to iterate,
        # from span (0, w) to span (n, n+w) given width w
        n = seq_len - w
        # ilr = C(i->r) + C(j->r+1)
        # [n, w, batch_size]
        ilr = stripe(s_c, n, w) + stripe(s_c, n, w, (w, 1))
        if ilr.requires_grad:
            ilr.register_hook(lambda x: x.masked_fill_(torch.isnan(x), 0))

        if viterbi:
            il = ir = ilr.permute(2, 0, 1).max(-1)[0]
        else:
            il = ir = ilr.permute(2, 0, 1).logsumexp(-1)

        # I(j->i) = logsumexp(C(i->r) + C(j->r+1)) + s(j->i), i <= r < j
        # fill the w-th diagonal of the lower triangular part of s_i
        # with I(j->i) of n spans
        s_i.diagonal(-w).copy_(il + scores.diagonal(-w))
        # I(i->j) = logsumexp(C(i->r) + C(j->r+1)) + s(i->j), i <= r < j
        # fill the w-th diagonal of the upper triangular part of s_i
        # with I(i->j) of n spans
        s_i.diagonal(w).copy_(ir + scores.diagonal(w))

        # C(j->i) = logsumexp(C(r->i) + I(j->r)), i <= r < j
        cl = stripe(s_c, n, w, (0, 0), 0) + stripe(s_i, n, w, (w, 0))
        if cl.requires_grad:
            cl.register_hook(lambda x: x.masked_fill_(torch.isnan(x), 0))

        if viterbi:
            s_c.diagonal(-w).copy_(cl.permute(2, 0, 1).max(-1)[0])
        else:
            s_c.diagonal(-w).copy_(cl.permute(2, 0, 1).logsumexp(-1))

        # C(i->j) = logsumexp(I(i->r) + C(r->j)), i < r <= j

        cr = stripe(s_i, n, w, (0, 1)) + stripe(s_c, n, w, (1, w), 0)
        if cr.requires_grad:
            cr.register_hook(lambda x: x.masked_fill_(torch.isnan(x), 0))

        if viterbi:
            s_c.diagonal(w).copy_(cr.permute(2, 0, 1).max(-1)[0])
        else:
            s_c.diagonal(w).copy_(cr.permute(2, 0, 1).logsumexp(-1))

        # disable multi words to modify the root
        s_c[0, w][lens.ne(w)] = float('-inf')

    logZ = s_c[0].gather(0, lens.unsqueeze(0))

    if not decode and not max_margin:
        return logZ

    logZ.sum().backward()

    if decode:
        dep = scores_origin.grad
        predicted_arc = dep.new_zeros(dep.shape[0], dep.shape[1]).long()
        arc = dep.nonzero()
        predicted_arc[arc[:, 0], arc[:, 1]] = arc[:, 2]
        ctx['arc_pred'] = predicted_arc

    if max_margin:
        ctx['s_arc_grad'] = scores_origin.grad

    return



# # https://github.com/sustcsonglin/second-order-neural-dmv/blob/main/parser/dmvs/dmv1o.py
@torch.enable_grad()
def eisner_headsplit(ctx, decode=False, max_margin=False):
    attach = ctx['s_arc']
    lens = ctx['seq_len']

    batch_size, seq_len = attach.shape[:2]

    if max_margin:
        augment_score(ctx)

    if 's_bd' in ctx:
        span_bd_left_score = ctx['s_bd']
        span_bd_right_score = span_bd_left_score

    else:
        span_bd_left_score = ctx['s_bd_left']
        span_bd_right_score = ctx['s_bd_right']

    if decode or max_margin:
        attach = attach.detach().clone().requires_grad_(True)
        span_bd_left_score = span_bd_left_score.detach().clone().requires_grad_(True)
        span_bd_right_score = span_bd_right_score.detach().clone().requires_grad_(True)

    b, N, *_ = attach.shape
    n = N - 1

    A = 0
    B = 1
    L = 0
    R = 1
    HASCHILD = 1
    NOCHILD = 0

    viterbi = max_margin or decode

    def get_plus_semiring():
        if viterbi:
            def plus(x, dim):
                return torch.max(x, dim)[0]
        else:
            def plus(x, dim):
                return torch.logsumexp(x, dim)
        return plus

    semiring_plus = get_plus_semiring()

    alpha_C = [
        [attach.new_zeros(b, N, N, 2).fill_(-1e9) for _ in range(2)] for _ in range(2)
    ]

    # incomplete spans.
    alpha_I = [
        [attach.new_zeros(b, N, N).fill_(-1e9) for _ in range(2)] for _ in range(2)
    ]

    # handling root
    # alpha_C[A][R][:, 0, 0, HASCHILD] = 0
    # alpha_C[B][R][:, 0, -1, HASCHILD] = 0

    # initialize closed spans
    alpha_C[A][R][:, 1:, 0, NOCHILD] = span_bd_right_score[:, torch.arange(n), torch.arange(n) + 1]
    alpha_C[B][R][:, 1:, -1, NOCHILD] = span_bd_right_score[:, torch.arange(n), torch.arange(n) + 1]

    alpha_C[A][L][:, 1:, 0, NOCHILD] = span_bd_left_score[:, torch.arange(n), torch.arange(n)]
    alpha_C[B][L][:, 1:, -1, NOCHILD] = span_bd_left_score[:, torch.arange(n), torch.arange(n)]

    # initialize open span
    alpha_C[A][R][:, :, 0, HASCHILD] = 0
    alpha_C[B][R][:, :, -1, HASCHILD] = 0
    alpha_C[A][L][:, :, 0, HASCHILD] = 0
    alpha_C[B][L][:, :, -1, HASCHILD] = 0

    # single root.
    start_idx = 1
    for k in range(1, N - start_idx):
        f = torch.arange(start_idx, N - k), torch.arange(k + start_idx, N)
        ACL = alpha_C[A][L][:, start_idx: N - k, :k]
        ACR = alpha_C[A][R][:, start_idx: N - k, :k]
        BCL = alpha_C[B][L][:, start_idx + k:, N - k:]
        BCR = alpha_C[B][R][:, start_idx + k:, N - k:]
        x = semiring_plus(ACR[..., NOCHILD] + BCL[..., HASCHILD], dim=2)
        arcs_l = x + attach[:, f[0], f[1]]
        alpha_I[A][L][:, start_idx: N - k, k] = arcs_l
        alpha_I[B][L][:, k + start_idx:N, N - k - 1] = arcs_l
        x = semiring_plus(ACR[..., HASCHILD] + BCL[..., NOCHILD], dim=2)
        arcs_r = x + attach[:, f[1], f[0]]
        alpha_I[A][R][:, start_idx: N - k, k] = arcs_r
        alpha_I[B][R][:, k + start_idx:N, N - k - 1] = arcs_r

        AIR = alpha_I[A][R][:, start_idx: N - k, 1: k + 1]
        BIL = alpha_I[B][L][:, k + start_idx:, N - k - 1: N - 1]
        new = semiring_plus(ACL[..., NOCHILD] + BIL, dim=2)
        new2 = new + span_bd_left_score[:, torch.arange(k, N - 1), torch.arange(0, N - k - 1)]

        alpha_C[A][L][:, start_idx: N - k, k, HASCHILD] = new
        alpha_C[A][L][:, start_idx: N - k, k, NOCHILD] = new2
        alpha_C[B][L][:, k + start_idx:N, N - k - 1, HASCHILD] = new
        alpha_C[B][L][:, k + start_idx:N, N - k - 1, NOCHILD] = new2

        new = semiring_plus(AIR + BCR[..., NOCHILD], dim=2)
        new2 = new + span_bd_right_score[:, torch.arange(N - k - 1), torch.arange(k + 1, N)]
        alpha_C[A][R][:, start_idx:N - k, k, HASCHILD] = new
        alpha_C[A][R][:, start_idx:N - k, k, NOCHILD] = new2
        alpha_C[B][R][:, start_idx + k:N, N - k - 1, HASCHILD] = new
        alpha_C[B][R][:, start_idx + k:N, N - k - 1, NOCHILD] = new2

    # dealing with the root.
    root_incomplete_span = alpha_C[A][L][:, 1, :N - 1, NOCHILD] + attach[:, 1:, 0]
    for k in range(1, N):
        AIR = root_incomplete_span[:, :k]
        BCR = alpha_C[B][R][:, k, N - k:, NOCHILD]
        alpha_C[A][R][:, 0, k, NOCHILD] = semiring_plus(AIR + BCR, dim=1)

    logZ = alpha_C[A][R][torch.arange(batch_size), 0, lens, NOCHILD]

    # CRF loss.
    if not decode and not max_margin:
        return logZ

    logZ.sum().backward()

    if decode:
        predicted_arc = span_bd_left_score.new_zeros(batch_size, seq_len).long()
        arc = attach.grad.nonzero()
        predicted_arc[arc[:, 0], arc[:, 1]] = arc[:, 2]
        ctx['arc_pred'] = predicted_arc

    if max_margin:
        ctx['s_arc_grad'] = attach.grad
        if 's_bd' in ctx:
            ctx['s_bd_grad'] = span_bd_left_score.grad
        else:
            ctx['s_bd_left_grad'] = span_bd_left_score.grad
            ctx['s_bd_right_grad'] = span_bd_right_score.grad
            # assert span_bd_left_score.grad.sum() == span_bd_right_score.grad.sum() == ctx['seq_len'].sum()
    return logZ



