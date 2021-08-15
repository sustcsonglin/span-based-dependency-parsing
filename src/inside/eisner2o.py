import torch
from supar.utils.fn import *
from src.inside.eisner import eisner
from src.loss.get_score import augment_score
from torch.utils.checkpoint import checkpoint


@torch.enable_grad()
def eisner2o(ctx, decode=False,  max_margin=False):
    if max_margin:
        augment_score(ctx)

    s_arc_origin = ctx['s_arc']
    s_sib_origin = ctx['s_sib']

    lens = ctx['seq_len']


    s_arc_origin = s_arc_origin.clone().detach().requires_grad_(True)
    s_sib_origin = s_sib_origin.clone().detach().requires_grad_(True)

    batch_size, seq_len, _ = s_arc_origin.shape
    # [seq_len, seq_len, batch_size]
    s_arc = s_arc_origin.permute(2, 1, 0)
    # [seq_len, seq_len, seq_len, batch_size]

    s_sib = s_sib_origin.permute(2, 1, 3, 0)
    s_i = torch.full_like(s_arc, float('-inf'))
    s_s = torch.full_like(s_arc, float('-inf'))
    s_c = torch.full_like(s_arc, float('-inf'))
    s_c.diagonal().fill_(0)
    
    viterbi = decode or max_margin

    for w in range(1, seq_len):
        # n denotes the number of spans to iterate,
        # from span (0, w) to span (n, n+w) given width w
        n = seq_len - w
        # I(j->i) = logsum(exp(I(j->r) + S(j->r, i)) +, i < r < j
        #                  exp(C(j->j) + C(i->j-1)))
        #           + s(j->i)
        # [n, w, batch_size]

        il = stripe(s_i, n, w, (w, 1)) + stripe(s_s, n, w, (1, 0), 0)
        il += stripe(s_sib[range(w, n+w), range(n)], n, w, (0, 1))
        # [n, 1, batch_size]
        il0 = stripe(s_c, n, 1, (w, w)) + stripe(s_c, n, 1, (0, w - 1))
        # il0[0] are set to zeros since the scores of the complete spans starting from 0 are always -inf
        il[:, -1] = il0.index_fill_(0, lens.new_tensor(0), 0).squeeze(1)

        if il.requires_grad:
            il.register_hook(lambda x: x.masked_fill_(torch.isnan(x), 0))

        if not viterbi:
            il = il.permute(2, 0, 1).logsumexp(-1)
        else:
            il = il.permute(2, 0, 1).max(-1)[0]

        s_i.diagonal(-w).copy_(il + s_arc.diagonal(-w))
        # I(i->j) = logsum(exp(I(i->r) + S(i->r, j)) +, i < r < j
        #                  exp(C(i->i) + C(j->i+1)))
        #           + s(i->j)
        # [n, w, batch_size]
        ir = stripe(s_i, n, w) + stripe(s_s, n, w, (0, w), 0)
        ir += stripe(s_sib[range(n), range(w, n+w)], n, w)
        ir[0] = float('-inf')
        # [n, 1, batch_size]

        ir0 = stripe(s_c, n, 1) + stripe(s_c, n, 1, (w, 1))
        ir[:, 0] = ir0.squeeze(1)
        if ir.requires_grad:
            ir.register_hook(lambda x: x.masked_fill_(torch.isnan(x), 0))

        if not viterbi:
            ir = ir.permute(2, 0, 1).logsumexp(-1)
        else:
            ir = ir.permute(2, 0, 1).max(-1)[0]

        s_i.diagonal(w).copy_(ir + s_arc.diagonal(w))
        # [n, w, batch_size]
        slr = stripe(s_c, n, w) + stripe(s_c, n, w, (w, 1))
        if slr.requires_grad:
            slr.register_hook(lambda x: x.masked_fill_(torch.isnan(x), 0))

        if not viterbi:
            slr = slr.permute(2, 0, 1).logsumexp(-1)

        else:
            slr = slr.permute(2, 0, 1).max(-1)[0]


        # S(j, i) = logsumexp(C(i->r) + C(j->r+1)), i <= r < j
        s_s.diagonal(-w).copy_(slr)
        # S(i, j) = logsumexp(C(i->r) + C(j->r+1)), i <= r < j
        s_s.diagonal(w).copy_(slr)
        # C(j->i) = logsumexp(C(r->i) + I(j->r)), i <= r < j
        cl = stripe(s_c, n, w, (0, 0), 0) + stripe(s_i, n, w, (w, 0))
        cl.register_hook(lambda x: x.masked_fill_(torch.isnan(x), 0))

        if not viterbi:
            s_c.diagonal(-w).copy_(cl.permute(2, 0, 1).logsumexp(-1))
        else:
            s_c.diagonal(-w).copy_(cl.permute(2, 0, 1).max(-1)[0])

        # C(i->j) = logsumexp(I(i->r) + C(r->j)), i < r <= j
        cr = stripe(s_i, n, w, (0, 1)) + stripe(s_c, n, w, (1, w), 0)
        cr.register_hook(lambda x: x.masked_fill_(torch.isnan(x), 0))

        if not viterbi:
            s_c.diagonal(w).copy_(cr.permute(2, 0, 1).logsumexp(-1))
        else:
            s_c.diagonal(w).copy_(cr.permute(2, 0, 1).max(-1)[0])
        # disable multi words to modify the root
        s_c[0, w][lens.ne(w)] = float('-inf')

    logZ = s_c[0].gather(0, lens.unsqueeze(0))

    if not (decode or max_margin):
        return logZ

    logZ.sum().backward()

    if decode:
        dep = s_arc_origin.grad
        predicted_arc = dep.new_zeros(dep.shape[0], dep.shape[1]).long()
        arc = dep.nonzero()
        predicted_arc[arc[:, 0], arc[:, 1]] = arc[:, 2]
        ctx['arc_pred'] = predicted_arc

    if max_margin:
        ctx['s_arc_grad'] = s_arc_origin.grad
        ctx['s_sib_grad'] = s_sib_origin.grad






@torch.enable_grad()
def eisner2o_headsplit(ctx, decode=False, max_margin=False):
    attach = ctx['s_arc']
    s_sib = ctx['s_sib']

    lens = ctx['seq_len']
    batch_size, seq_len = attach.shape[:2]

    if max_margin:
        augment_score(ctx)

    span_bd_left_score = ctx['s_bd_left']
    span_bd_right_score = ctx['s_bd_right']

    if decode or max_margin:
        attach = attach.detach().clone().requires_grad_(True)
        span_bd_left_score = span_bd_left_score.detach().clone().requires_grad_(True)
        span_bd_right_score = span_bd_right_score.detach().clone().requires_grad_(True)
        s_sib = s_sib.detach().clone().requires_grad_(True)

    b, N, *_ = attach.shape

    # s_sib
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

    #
    alpha_S = [attach.new_zeros(b, N, N).fill_(-1e9)
    for _ in range(2)]



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

        AIR = alpha_I[A][R][:, start_idx: N-k, :k]
        BIL = alpha_I[B][L][:, start_idx +k:, N-k:]

        x = semiring_plus(ACR[..., NOCHILD] + BCL[..., NOCHILD], dim=2)
        alpha_S[A][:, start_idx: N-k, k] = x
        alpha_S[B][:, k+ start_idx:N, N-k-1] = x

        AS = alpha_S[A][:, start_idx:N-k, 1:k]
        BS = alpha_S[B][:, start_idx+k:, N-k:-1]

        n = ACL.shape[1]
        w = seq_len - n
        x2 = ACR[..., -1, NOCHILD, ] + BCL[..., -1, HASCHILD]


        def op1(a, b, c):
            return semiring_plus(a + b + c[:, :, :-1], dim=2)

        def op2(a, b, c):
            return semiring_plus(a + b + c[:, :, 1:], dim=2)



        if k > 1:
            x1 = checkpoint(op1, AS.clone(), stripe_sib_left(s_sib, n, w), BIL.clone())
            x = semiring_plus(torch.stack([x1, x2]), dim=0)
        else:
            x = x2

        arcs_l = x + attach[:, f[0], f[1]]
        alpha_I[A][L][:, start_idx: N - k, k] = arcs_l
        alpha_I[B][L][:, k + start_idx:N, N - k - 1] = arcs_l

        x2 = ACR[..., 0, HASCHILD] + BCL[..., 0, NOCHILD]
        if k > 1:
            x1 = checkpoint(op2, BS.clone(), stripe_sib_right(s_sib, n, w), AIR.clone())
            x = semiring_plus(torch.stack([x1, x2]), dim=0)
        else:
            x = x2

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
        ctx['s_sib_grad'] = s_sib.grad
        ctx['s_bd_left_grad'] = span_bd_left_score.grad
        ctx['s_bd_right_grad'] = span_bd_right_score.grad
    return logZ


def stripe_sib_left(x, n, w):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2] + stride[3])
    new_stride.append(stride[3])
    return x.as_strided(size=(x.shape[0], n, w-2),
                        stride=new_stride,
                        storage_offset=stride[1] + (w)*stride[2] + 2*stride[3])

def stripe_sib_right(x, n, w):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2] + stride[3])
    new_stride.append(stride[3])
    return x.as_strided(size=(x.shape[0], n, w-2),
                        stride=new_stride,
                        storage_offset= w*stride[1] + stride[2] + 2*stride[3])


