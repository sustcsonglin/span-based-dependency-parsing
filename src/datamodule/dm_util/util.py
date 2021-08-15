import unicodedata

punct_set = '.' '``' "''" ':' ','
import re


# https://github.com/DoodleJZ/HPSG-Neural-Parser/blob/cdcffa78945359e14063326cadd93fd4c509c585/src_joint/dep_eval.py
def is_uni_punctuation(word):
    match = re.match("^[^\w\s]+$]", word, flags=re.UNICODE)
    return match is not None

def is_punctuation(word, pos, punct_set=punct_set):
    if punct_set is None:
        return is_uni_punctuation(word)
    else:
        return pos in punct_set or pos == 'PU' # for chinese

def get_path(path):
    return path

def get_path_debug(path):
    return path + ".debug"

def clean_number(w):
    new_w = re.sub('[0-9]{1,}([,.]?[0-9]*)*', '0', w)
    return new_w

def clean_word(words):
    PTB_UNESCAPE_MAPPING = {
        "«": '"',
        "»": '"',
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        "„": '"',
        "‹": "'",
        "›": "'",
        "\u2013": "--",  # en dash
        "\u2014": "--",  # em dash
    }
    cleaned_words = []
    for word in words:
        word = PTB_UNESCAPE_MAPPING.get(word, word)
        word = word.replace("\\/", "/").replace("\\*", "*")
        # Mid-token punctuation occurs in biomedical text
        word = word.replace("-LSB-", "[").replace("-RSB-", "]")
        word = word.replace("-LRB-", "(").replace("-RRB-", ")")
        word = word.replace("-LCB-", "{").replace("-RCB-", "}")
        word = word.replace("``", '"').replace("`", "'").replace("''", '"')
        word = clean_number(word)
        cleaned_words.append(word)
    return cleaned_words




def find_dep_boundary(heads):
    left_bd = [i for i in range(len(heads))]
    right_bd = [i + 1  for i in range(len(heads))]

    for child_idx, head_idx in enumerate(heads):
        if head_idx > 0:
            if left_bd[child_idx] < left_bd[head_idx - 1]:
                left_bd[head_idx - 1] = left_bd[child_idx]

            elif child_idx > right_bd[head_idx - 1] - 1:
                right_bd[head_idx - 1] = child_idx + 1
                while head_idx != 0:
                    if heads[head_idx-1] > 0 and  child_idx + 1 > right_bd[ heads[head_idx-1] - 1] :
                        right_bd[ heads[head_idx-1] - 1]  =  child_idx + 1
                        head_idx = heads[head_idx-1]
                    else:
                        break

    # (head_word_idx, left_bd_idx, right_bd_idx)
    triplet = []
    # head index should add1, as the root token would be the first token.
    # [ )  left bdr, right bdr.
    # for i in range(len(heads)):
    # what do I want?
    # 生成整个span的score????????
    # seems ok.s

    # left boundary, right boundary, parent, head
    for i, (parent, left_bdr, right_bdr) in enumerate(zip(heads, left_bd, right_bd)):
        triplet.append([left_bdr, right_bdr, parent-1, i])





    return triplet





def isProjective(heads):
    pairs = [(h, d) for d, h in enumerate(heads, 1) if h >= 0]
    for i, (hi, di) in enumerate(pairs):
        for hj, dj in pairs[i+1:]:
            (li, ri), (lj, rj) = sorted([hi, di]), sorted([hj, dj])
            if li <= hj <= ri and hi == dj:
                return False
            if lj <= hi <= rj and hj == di:
                return False
            if (li < lj < ri or li < rj < ri) and (li - lj)*(ri - rj) > 0:
                return False
    return True






