# obtain the score of unlabeled trees.
def u_score(ctx):
    score = 0

    if 's_arc' in ctx:
        score += get_arc_score(ctx)

    if 's_span_head_word' in ctx:
        score += get_span_head_word_score(ctx)

    if 's_bd' in ctx:
        score += get_bd_score(ctx)

    if 's_bd_left' in ctx:
        assert 's_bd_right' in ctx
        score += get_bd_left_score(ctx)
        score += get_bd_right_score(ctx)

    if 's_sib' in ctx:
        score += get_sib_score(ctx)

    return score

def predict_score_mm(ctx):
    score = 0
    if 's_arc' in ctx :
        score += (ctx['s_arc'] * ctx['s_arc_grad']).sum()

    if 's_span_head_word' in ctx:
        score += (ctx['s_span_head_word'] * ctx['s_span_head_word_grad']).sum()

    if 's_bd' in ctx:
        score += (ctx['s_bd'] * ctx['s_bd_grad']).sum()

    if 's_bd_left' in ctx:
        score += (ctx['s_bd_left'] * ctx['s_bd_left_grad']).sum()

    if 's_bd_right' in ctx:
        score += (ctx['s_bd_right'] * ctx['s_bd_right_grad']).sum()

    if 's_sib' in ctx:
        try:
            score += (ctx['s_sib'] * ctx['s_sib_grad']).sum()
        except:
            # corner case: (e.g. sentences of length 2, no siblings)
            pass

    return score


def augment_score(ctx):
    if 's_arc' in ctx:
        aug_arc_score(ctx)

    if 's_span_head_word' in ctx:
        aug_span_head_word_score(ctx)

    if 's_bd' in ctx:
        aug_bd_score(ctx)

    if 's_bd_left' in ctx:
        aug_bd_left_score(ctx)

    if 's_bd_right' in ctx:
        aug_bd_right_score(ctx)

    if 's_sib' in ctx:
        aug_sib_score(ctx)



def get_arc_score(ctx):
    s_arc = ctx['s_arc']
    arcs = ctx['head']
    return s_arc[arcs[:, 0], arcs[:, 1], arcs[:, 2]].sum()



def aug_arc_score(ctx):
    s_arc = ctx['s_arc']
    arcs = ctx['head']
    s_arc[arcs[:, 0], arcs[:, 1], arcs[:, 2]] -= 1



def get_sib_score(ctx):
    s_sib = ctx['s_sib']
    sib = ctx['sib']
    try:
        return s_sib[sib[:, 0], sib[:, 1], sib[:, 2], sib[:, 3]].sum()
    except:
        return 0

def aug_sib_score(ctx):
    s_sib = ctx['s_sib']
    sib = ctx['sib']
    try:
        s_sib[sib[:, 0], sib[:, 1], sib[:, 2], sib[:, 3]] -= 1
    except:
        pass


def get_span_head_word_score(ctx):
    span_head_word = ctx['span_head_word']
    s_span_head_word = ctx['s_span_head_word']
    score = s_span_head_word[span_head_word[:, 0], span_head_word[:, 1], span_head_word[:, 2], span_head_word[:, 3]].sum()
    return score

def aug_span_head_word_score(ctx):
    span_head_word = ctx['span_head_word']
    s_span_head_word = ctx['s_span_head_word']
    s_span_head_word[span_head_word[:, 0], span_head_word[:, 1], span_head_word[:, 2], span_head_word[:, 3]] -= 1.5


def get_bd_score(ctx):
    s_bd = ctx['s_bd']
    span_head = ctx['span_head_word']
    score = 0
    score += s_bd[span_head[:, 0], span_head[:, 3], span_head[:, 1]].sum()
    score += s_bd[span_head[:, 0], span_head[:, 3], span_head[:, 2]].sum()
    return score

def aug_bd_score(ctx):
    s_bd = ctx['s_bd']
    span_head = ctx['span_head_word']
    s_bd[span_head[:, 0], span_head[:, 3], span_head[:,1]] -= 1
    s_bd[span_head[:, 0], span_head[:, 3], span_head[:,2]] -= 1


def get_bd_left_score(ctx):
    s_bd = ctx['s_bd_left']
    span_head = ctx['span_head_word']
    score = s_bd[span_head[:, 0], span_head[:, 3], span_head[:, 1]].sum()
    return score

def aug_bd_left_score(ctx):
    s_bd = ctx['s_bd_left']
    span_head = ctx['span_head_word']
    s_bd[span_head[:, 0], span_head[:, 3], span_head[:,1]] -= 1


def get_bd_right_score(ctx):
    s_bd = ctx['s_bd_right']
    span_head = ctx['span_head_word']
    score = s_bd[span_head[:, 0], span_head[:, 3], span_head[:, 2]].sum()
    return score


def aug_bd_right_score(ctx):
    s_bd = ctx['s_bd_right']
    span_head = ctx['span_head_word']
    s_bd[span_head[:, 0], span_head[:, 3], span_head[:, 2]] -= 1





