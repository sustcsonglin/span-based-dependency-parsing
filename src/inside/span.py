import torch
from src.inside.fn import *
from src.inside.eisner_satta import es4dep
from ..loss.get_score import augment_score

# O(n3) algorithm, only use span core, do not use arc score for projective dependency parsing.
@torch.enable_grad()
def span_inside(ctx, decode=False, max_margin=False):

    assert decode or max_margin

    if max_margin:
        augment_score(ctx)

    s_span_score = ctx['s_span_head_word']
    lens = ctx['seq_len']

    s_span_score = s_span_score.detach().clone().requires_grad_(True)

    b, seq_len = s_span_score.shape[:2]

    s_inside_children = s_span_score.new_zeros(b, seq_len, seq_len).fill_(-1e9)
    s_inside_close = s_span_score.new_zeros(b, seq_len, seq_len).fill_(-1e9)

    # do I need s_close? it seems that i do not need this term? right
    s_inside_children[:, torch.arange(seq_len-1), torch.arange(seq_len-1)+1] = s_span_score[:, torch.arange(seq_len-1), torch.arange(seq_len-1)+1, torch.arange(seq_len-1)]

    for w in range(2, seq_len):
        n = seq_len - w

        # two child compose together.
        left = stripe(s_inside_children, n, w - 1, (0, 1))
        right = stripe(s_inside_children, n, w - 1, (1, w), 0)
        compose = (left + right).max(2)[0]

        # case 1: the head word is right-most
        l = left[:, :, -1]
        compose_score1 = l + s_span_score[:, torch.arange(n), torch.arange(n)+w, torch.arange(n)+w-1]

        # case 2: the head word is left-most.
        r = right[:, :, 0]
        compose_score2 = r + s_span_score[:, torch.arange(n), torch.arange(n)+w, torch.arange(n)]

        if w > 2:
            left = stripe(s_inside_children, n, w - 2, (0, 1))
            right = stripe(s_inside_children, n, w - 2, (2, w), 0)
            compose_score3 = left + right + diagonal_v2(s_span_score, w)[:, :, 1:-1]
            compose_score = torch.cat([compose_score1.unsqueeze(2), compose_score3, compose_score2.unsqueeze(2)], dim=2)
            compose_score = compose_score.max(2)[0]

        else:
            compose_score = torch.cat([compose_score1.unsqueeze(2), compose_score2.unsqueeze(2)], dim=2)
            compose_score = compose_score.max(2)[0]

        compose = torch.stack([compose, compose_score]).max(0)[0]
        diagonal_copy_(s_inside_children, compose, w)
        diagonal_copy_(s_inside_close, compose_score, w)

    try:
        s_inside_close[torch.arange(b), 0, lens].sum().backward()
    except:
        pass

    if decode:
       ctx['arc_pred'] = recover_arc(s_span_score.grad, lens)

    if max_margin:
        ctx['s_span_head_word_grad'] = s_span_score.grad




# heads: (left, right, head word).
# from decoded span to recover arcs.
def recover_arc(heads, lens):
    if heads is None:
        return lens.new_zeros(lens.shape[0], lens.max() + 1)
    result = np.zeros(shape=(heads.shape[0], heads.shape[1]))
    lens = lens.detach().cpu().numpy()
    for i in range(heads.shape[0]):
        if lens[i] == 1:
            result[i][1] = 0
        else:
            span = heads[i].detach().nonzero().cpu().numpy()
            start = span[:,0]
            end = span[:,1]
            preorder_sort = np.lexsort((-end, start))
            start = start[preorder_sort]
            end = end[preorder_sort]
            head = span[:,2][preorder_sort]
            stack = []
            arcs = []
            stack.append((start[0], end[0], 0))
            result[i][head[0]+1] = 0
            j = 0
            while j < start.shape[0]-1:
                j+=1
                s = start[j]
                e = end[j]
                top = stack[-1]
                top_s, top_e, top_i = top
                if top_s <= s and top_e >= e:
                    arcs.append((head[top_i] + 1, head[j] + 1))
                    result[i][head[j] + 1] = head[top_i] + 1
                    stack.append((s, e, j))
                else:
                    while top_s > s or top_e < e:
                        stack.pop()
                        top = stack[-1]
                        top_s, top_e, top_i = top
                    arcs.append([head[top_i] + 1, head[j] + 1])
                    result[i][head[j] + 1] = head[top_i] + 1
                    stack.append((s, e, j))
    return torch.tensor(result, device=heads.device, dtype=torch.long)











