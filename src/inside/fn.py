import torch
import numpy as np


def get_pre_order_span(predicted_span):
    predicted_span = predicted_span.detach().cpu().numpy()

    starts = []
    ends = []

    for i in range(predicted_span.shape[0]):
        span = predicted_span[i].nonzero()
        start = span[0]
        end = span[1]
        preorder_sort = np.lexsort((-end, start))
        starts.append(start[preorder_sort])
        ends.append(end[preorder_sort])

    return {'start_idx': starts,
            'end_idx': ends,
            }



# input: batch * child * head * nt * nt * nt
# output: batch * (n-1) * (w-1) * w * nt * nt * nt
def stripe_need_left_parent(x, n, w, offset=(0, 0), dim=1):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    numel = stride[2]
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append((seq_len + 1) * numel)
    new_stride.extend(*stride[2:])
    return x.as_strided(size=(x.shape[0], n, *list(x.shape[3:])),
                        stride=stride,
                        storage_offset=(offset[0] * seq_len + offset[1]) * numel)


def stripe_left(x, n, w, offset=(0, 0), dim=1):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    numel = stride[2]
    # stride[1] = (seq_len + 1) * numel
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append((seq_len + 1) * numel)
    new_stride.extend(*stride[2:])
    return x.as_strided(size=(x.shape[0], n, *list(x.shape[3:])),
                        stride=stride,
                        storage_offset=(offset[0] * seq_len + offset[1]) * numel)


def stripe_right(x, n, w, offset=(0, 0), dim=1):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    numel = stride[2]
    # stride[1] = (seq_len + 1) * numel
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append((seq_len + 1) * numel)
    new_stride.extend(*stride[2:])

    return x.as_strided(size=(x.shape[0], n, *list(x.shape[3:])),
                        stride=stride,
                        storage_offset=(offset[0] * seq_len + offset[1]) * numel) +  (w-1) * numel * (
               1 if dim == 0 else seq_len)





def stripe(x, n, w, offset=(0, 0), dim=1):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    numel = stride[2]
    stride[1] = (seq_len + 1) * numel
    stride[2] = (1 if dim == 1 else seq_len) * numel
    if len(x.shape) > 3:
        return x.as_strided(size=(x.shape[0], n, w, *list(x.shape[3:])),
                            stride=stride,
                            storage_offset=(offset[0] * seq_len + offset[1]) * numel)
    else:
        return x.as_strided(size=(x.shape[0], n, w),
                            stride=stride,
                            storage_offset=(offset[0] * seq_len + offset[1]) * numel)

def stripe_logadd1(x, value,  n, w,  offset=(0, 0), dim=1):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    numel = stride[2]
    stride[1] = (seq_len + 1) * numel
    stride[2] = (1 if dim == 1 else seq_len) * numel
    if len(x.shape) > 3:
        tmp = x.as_strided(size=(x.shape[0], n, w, *list(x.shape[3:])),
                            stride=stride,
                            storage_offset=(offset[0] * seq_len + offset[1]) * numel)
        x.as_strided(size=(x.shape[0], n, w, *list(x.shape[3:])),
                     stride=stride,
                     storage_offset=(offset[0] * seq_len + offset[1] + 1) * numel).copy_(torch.logaddexp(tmp, value))
    else:
        raise NotImplemented
        return x.as_strided(size=(x.shape[0], n, w),
                            stride=stride,
                            storage_offset=(offset[0] * seq_len + offset[1]) * numel)




def logbmmexp(x, y):
    x = x.contiguous()
    y = y.contiguous()
    return (x.unsqueeze(-1) + y.unsqueeze(-3)).logsumexp(-2)

def maxbmm(x, y):
    return (x.unsqueeze(-1) + y.unsqueeze(-3)).max(-2)[0]





def stripe_grammar_rules(x, n, w, offset=0, addition = 0):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1])
    new_stride.append(stride[1])
    new_stride.extend(stride[2:])
    return x.as_strided(size=(x.shape[0], n, w, *x.shape[2:]),
                            stride=new_stride, storage_offset= offset*(stride[1]))






def stripe_version3(x, n, w, offset=0):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1]+stride[2])
    new_stride.append(stride[1])
    new_stride.extend(stride[2:])
    return x.as_strided(size=(x.shape[0], n, w, w, *list(x.shape[3:])),
                            stride=new_stride,
                            storage_offset=0)

def stripe_version5(x, n, w=0):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1])
    new_stride.append(stride[1])
    new_stride.extend(stride[2:])
    return x.as_strided(size=(x.shape[0], n, w, *list(x.shape[2:])),
                            stride=new_stride,
                            storage_offset=0)

def stripe_span_head(x, n, w):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2])
    new_stride.append(stride[2])
    new_stride.extend(stride[3:])
    return x.as_strided(size=(x.shape[0], n, w, *list(x.shape[3:])),
                            stride=new_stride,
                            storage_offset=0)



def stripe_parent_left(x, n, w, nt, t):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2])
    new_stride.append(stride[2])
    new_stride.extend(stride[3:])
    return x.as_strided(size=(x.shape[0], n,  w,  x.shape[3], t, nt),
                            stride=new_stride,
                            storage_offset=stride[2] + stride[-2]*nt)

def stripe_parent_left_add(x, y, n, w, nt, t):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2])
    new_stride.append(stride[2])
    new_stride.extend(stride[3:])
    x.as_strided(size=(x.shape[0], n, w, x.shape[3], t, nt),
                 stride=new_stride,
                 storage_offset=stride[2] + stride[-2] * nt).add_(y)

def stripe_headed_left(x, n, w, nt, t):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2])
    new_stride.append(stride[1])
    new_stride.extend(stride[3:])
    return x.as_strided(size=(x.shape[0], n, w, x.shape[3], nt,  t),
                            stride=new_stride,
                            storage_offset= w * stride[2] + nt * stride[-1])


def stripe_rules_left(x, w, start, nt):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1])
    new_stride.append(stride[2])
    new_stride.extend(stride[3:])
    return x.as_strided(size=(x.shape[0], w, start, nt, nt, nt),
                            stride=new_stride,
                            storage_offset= start * stride[1])


def stripe_rules_right(x, w, end, nt):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1])
    new_stride.append(stride[2])
    new_stride.extend(stride[3:])
    return x.as_strided(size=(x.shape[0], w, seq_len - end , nt, nt, nt),
                            stride=new_stride,
                            storage_offset= end * stride[2])





def stripe_headed_left_add(x, y, n, w, nt, t):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2])
    new_stride.append(stride[1])
    new_stride.extend(stride[3:])
    x.as_strided(size=(x.shape[0], n, w, x.shape[3], nt, t),
                 stride=new_stride,
                 storage_offset=w * stride[2] + nt * stride[-1]).add_(y)
    # return x


def stripe_headed_right(x, n, w, nt, t):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2])
    new_stride.append(stride[1])
    new_stride.extend(stride[3:])
    return x.as_strided(size=(x.shape[0], n, w, x.shape[3], t, nt),
                            stride=new_stride,
                            storage_offset= stride[1] + nt*stride[-2])

def stripe_headed_right_add(x, y, n, w, nt, t):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2])
    new_stride.append(stride[1])
    new_stride.extend(stride[3:])
    x.as_strided(size=(x.shape[0], n, w, x.shape[3], t, nt),
                 stride=new_stride,
                 storage_offset=stride[1] + nt * stride[-2]).add_(y)

def stripe_parent_right(x, n, w, nt, t):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2])
    new_stride.append(stride[2])
    new_stride.extend(stride[3:])
    return x.as_strided(size=(x.shape[0], n, w, x.shape[3], nt, t),
                            stride=new_stride,
                            storage_offset=w*stride[1] + nt*stride[-1])

def stripe_parent_right_add(x, y, n, w, nt, t):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2])
    new_stride.append(stride[2])
    new_stride.extend(stride[3:])
    x.as_strided(size=(x.shape[0], n, w, x.shape[3], nt, t),
                     stride=new_stride,
                     storage_offset=w*stride[1] + nt*stride[-1]).add_(y)


# #
#
#
# def stripe_parent_left(x, n, w):
#     x, seq_len = x.contiguous(), x.size(2)
#     stride = list(x.stride())
#     new_stride = []
#     new_stride.append(stride[0])
#     new_stride.append(stride[1] + stride[2])
#     new_stride.append(stride[2])
#     new_stride.extend(stride[3:])
#     return x.as_strided(size=(x.shape[0], n,  w,  *list(x.shape[3:])),
#                             stride=new_stride,
#                             storage_offset=stride[2])
#
# def stripe_parent_left_add(x, y, n, w):
#     x, seq_len = x.contiguous(), x.size(2)
#     stride = list(x.stride())
#     new_stride = []
#     new_stride.append(stride[0])
#     new_stride.append(stride[1] + stride[2])
#     new_stride.append(stride[2])
#     new_stride.extend(stride[3:])
#     x.as_strided(size=(x.shape[0], n,  w,  *list(x.shape[3:])),
#                             stride=new_stride,
#                             storage_offset=stride[2]).add_(y)
#
#
# def stripe_headed_left(x, n, w):
#     x, seq_len = x.contiguous(), x.size(2)
#     stride = list(x.stride())
#     new_stride = []
#     new_stride.append(stride[0])
#     new_stride.append(stride[1] + stride[2])
#     new_stride.append(stride[1])
#     new_stride.extend(stride[3:])
#     return x.as_strided(size=(x.shape[0], n, w, *list(x.shape[3:])),
#                             stride=new_stride,
#                             storage_offset= w * stride[2])
#
# def stripe_headed_left_add(x, y, n, w):
#     x, seq_len = x.contiguous(), x.size(2)
#     stride = list(x.stride())
#     new_stride = []
#     new_stride.append(stride[0])
#     new_stride.append(stride[1] + stride[2])
#     new_stride.append(stride[1])
#     new_stride.extend(stride[3:])
#     x.as_strided(size=(x.shape[0], n, w, *list(x.shape[3:])),
#                             stride=new_stride,
#                             storage_offset= w * stride[2]).add_(y)
#
#
# def stripe_headed_right(x, n, w):
#     x, seq_len = x.contiguous(), x.size(2)
#     stride = list(x.stride())
#     new_stride = []
#     new_stride.append(stride[0])
#     new_stride.append(stride[1] + stride[2])
#     new_stride.append(stride[1])
#     new_stride.extend(stride[3:])
#     return x.as_strided(size=(x.shape[0], n, w, *list(x.shape[3:])),
#                             stride=new_stride,
#                             storage_offset= stride[1])
#
# def stripe_headed_right_add(x, y, n, w):
#     x, seq_len = x.contiguous(), x.size(2)
#     stride = list(x.stride())
#     new_stride = []
#     new_stride.append(stride[0])
#     new_stride.append(stride[1] + stride[2])
#     new_stride.append(stride[1])
#     new_stride.extend(stride[3:])
#     x.as_strided(size=(x.shape[0], n, w, *list(x.shape[3:])),
#                             stride=new_stride,
#                             storage_offset= stride[1]).add_(y)
#
#
#
# def stripe_parent_right(x, n, w):
#     x, seq_len = x.contiguous(), x.size(2)
#     stride = list(x.stride())
#     new_stride = []
#     new_stride.append(stride[0])
#     new_stride.append(stride[1] + stride[2])
#     new_stride.append(stride[2])
#     new_stride.extend(stride[3:])
#     return x.as_strided(size=(x.shape[0], n, w,  *list(x.shape[3:])),
#                             stride=new_stride,
#                             storage_offset=w*stride[1])
#
# def stripe_parent_right_add(x, y, n, w):
#     x, seq_len = x.contiguous(), x.size(2)
#     stride = list(x.stride())
#     new_stride = []
#     new_stride.append(stride[0])
#     new_stride.append(stride[1] + stride[2])
#     new_stride.append(stride[2])
#     new_stride.extend(stride[3:])
#     x.as_strided(size=(x.shape[0], n, w,  *list(x.shape[3:])),
#                             stride=new_stride,
#                             storage_offset=w*stride[1]).add_(y)











def stripe_version7(x, n, w=0):
    pass

def stripe_version6(x, n, w=0):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1]+stride[2])
    new_stride.append(stride[2])
    new_stride.append(stride[2])
    new_stride.extend(stride[2:])
    return x.as_strided(size=(x.shape[0], n, w, w, w,  *list(x.shape[3:])),
                            stride=new_stride,
                            storage_offset=0)



def decode_stripe1(x, n, w):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1]+stride[2]+stride[3])
    new_stride.append(stride[3])
    # new_stride.extend(stride[3:])
    return x.as_strided(size=(x.shape[0], n, w),
                            stride=new_stride,
                            storage_offset= (w) * (stride[2]))

def decode_stripe2(x, n, w):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1]+stride[2]+stride[3])
    new_stride.extend(stride[3:])
    return x.as_strided(size=(x.shape[0], n, w, *list(x.shape[4:])),
                            stride=new_stride,
                            storage_offset= w * stride[1])

def stripe_logadd_outside(x, y, n, w, offset):
    x = x.contiguous()
    stride =  list(x.stride())
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2] + stride[3])
    new_stride.append(stride[3])
    new_stride.extend(stride[4:])
    tmp = x.as_strided(size=(x.shape[0], n, w, *list(x.shape[4:])), stride=new_stride, storage_offset= offset * stride[1] + (offset + w) * stride[2] + offset * stride[3] )
    x.as_strided(size=(x.shape[0], n, w, *list(x.shape[4:])), stride=new_stride, storage_offset= offset * stride[1] + (offset + w) * stride[2] + offset * stride[3])\
        .copy_(torch.logaddexp(tmp, y))

def stripe_add_outside(x, y, n, w, offset):
    x = x.contiguous()
    stride =  list(x.stride())
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2] + stride[3])
    new_stride.append(stride[3])
    new_stride.extend(stride[4:])
    # tmp = x.as_strided(size=(x.shape[0], n, w, *list(x.shape[4:])), stride=new_stride, storage_offset= offset * stride[1] + (offset + w) * stride[2] + offset * stride[3] )
    x.as_strided(size=(x.shape[0], n, w, *list(x.shape[4:])), stride=new_stride, storage_offset= offset * stride[1] + (offset + w) * stride[2] + offset * stride[3])\
        .add_(y)

def stripe_need_dad_add(x, y,  n, w, start, end, headstart):
    x = x.contiguous()
    stride = list(x.stride())
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2] + stride[3])
    new_stride.append(stride[3])
    new_stride.extend(stride[4:])
    x.as_strided(size=(x.shape[0], n, w, *list(x.shape[4:])), stride=new_stride,
                       storage_offset= start * stride[1] + (end) * stride[2] + headstart * stride[3]).add_(y)



def stripe_need_dad(x, n, w, start, end, headstart):
    x = x.contiguous()
    stride = list(x.stride())
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2] + stride[3])
    new_stride.append(stride[3])
    new_stride.extend(stride[4:])
    return x.as_strided(size=(x.shape[0], n, w, *list(x.shape[4:])), stride=new_stride,
                       storage_offset= start * stride[1] + (end) * stride[2] + headstart * stride[3])


def stripe_need_child(x, n, w1, w2, start, end, headstart, childstart):
    x = x.contiguous()
    stride = list(x.stride())
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2] + stride[3] + stride[4])
    new_stride.append(stride[3])
    new_stride.extend(stride[4:])
    return x.as_strided(size=(x.shape[0], n, w1, w2, *list(x.shape[5:])), stride=new_stride,
                       storage_offset= start * stride[1] + (end) * stride[2] + headstart * stride[3] + childstart * stride[4])




def stripe_add_outside_v2(x, y, n, w, start, end, headstart):
    x = x.contiguous()
    stride = list(x.stride())
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2] + stride[3])
    new_stride.append(stride[3])
    new_stride.extend(stride[4:])
    # tmp = x.as_strided(size=(x.shape[0], n, w, *list(x.shape[4:])), stride=new_stride,
    #                    storage_offset= start * stride[1] + (end) * stride[2] + headstart * stride[3])
    x.as_strided(size=(x.shape[0], n, w, *list(x.shape[4:])), stride=new_stride,
                 storage_offset=start * stride[1] + (end) * stride[2] + headstart * stride[3]) \
        .add_(y)

def stripe_add_outside_left(x, y, n, w, start, end, headstart):
    x = x.contiguous()
    stride = list(x.stride())
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2])
    new_stride.append(stride[3])
    new_stride.extend(stride[4:])
    x.as_strided(size=(x.shape[0], n, w-1, *list(x.shape[4:])), stride=new_stride,
                 storage_offset=start * stride[1] + (end) * stride[2] + headstart * stride[3]) \
        .add_(y)

def stripe_add_outside_right(x, y, n, w, start, end, headstart):
    x = x.contiguous()
    stride = list(x.stride())
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2])
    new_stride.append(stride[3])
    new_stride.extend(stride[4:])
    x.as_strided(size=(x.shape[0], n, w-1, *list(x.shape[4:])), stride=new_stride,
                 storage_offset=start * stride[1] + (end) * stride[2] + headstart * stride[3]) \
        .add_(y)




def stripe_logadd_outside_v2(x, y, n, w, start, end, headstart):
    x = x.contiguous()
    stride = list(x.stride())
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2] + stride[3])
    new_stride.append(stride[3])
    new_stride.extend(stride[4:])
    tmp = x.as_strided(size=(x.shape[0], n, w, *list(x.shape[4:])), stride=new_stride,
                       storage_offset= start * stride[1] + (end) * stride[2] + headstart * stride[3])
    # print(tmp.shape)
    # print(y.shape)
    x.as_strided(size=(x.shape[0], n, w, *list(x.shape[4:])), stride=new_stride,
                 storage_offset=start * stride[1] + (end) * stride[2] + headstart * stride[3]) \
        .copy_(torch.logaddexp(tmp, y))


def stripe(x, n, w, offset=(0, 0), dim=1):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    numel = stride[2]
    stride[1] = (seq_len + 1) * numel
    stride[2] = (1 if dim == 1 else seq_len) * numel
    if len(x.shape) > 3:
        return x.as_strided(size=(x.shape[0], n, w, *list(x.shape[3:])),
                            stride=stride,
                            storage_offset=(offset[0] * seq_len + offset[1]) * numel)
    else:
        return x.as_strided(size=(x.shape[0], n, w),
                            stride=stride,
                            storage_offset=(offset[0] * seq_len + offset[1]) * numel)


def stripe_outside(x, y, n, w, offset):
    x = x.contiguous()
    stride =  list(x.stride())
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2] + stride[3])
    new_stride.append(stride[3])
    new_stride.extend(stride[4:])
    tmp = x.as_strided(size=(x.shape[0], n, w, *list(x.shape[4:])), stride=new_stride, storage_offset= offset * stride[1] + (offset + w) * stride[2] + offset * stride[3] )
    return tmp


def stripe_copy_gradient_left(x, y, n, left, right):
    x = x.contiguous()
    stride = list(x.stride())
    newstride = []
    newstride.append(stride[0])
    newstride.append(stride[1]+stride[2])
    # newstride.append(stride[1]+stride[2])
    newstride.append(stride[2])
    newstride.extend(stride[2:])
    tmp = x.as_strided(size=(x.shape[0], n, right, left, *x.shape[3:]), stride=newstride,
                      storage_offset=left*stride[2])
    print(tmp.shape)
    print(y.shape)
    x.as_strided(size=(x.shape[0], n, right, left, *x.shape[3:]), stride=newstride,
                storage_offset=right * stride[1]).copy_(torch.logaddexp(tmp, y))


def stripe_copy_gradient_right(x, y, n, left, right):
    x = x.contiguous()
    stride = list(x.stride())
    newstride = []
    newstride.append(stride[0])
    newstride.append(stride[1]+stride[2])
    # newstride.append(stride[1]+stride[2])
    newstride.append(stride[2])
    newstride.extend(stride[2:])
    tmp = x.as_strided(size=(x.shape[0], n, left, right, *x.shape[3:]), stride=newstride,
                      storage_offset=left*stride[2])
    # print(tmp.shape)
    # print(y.shape)
    x.as_strided(size=(x.shape[0], n, left, right, *x.shape[3:]), stride=newstride,
                storage_offset= right * stride[2]).copy_(torch.logaddexp(tmp, y))



# used in lexicalized-pcfg.
def stripe_version2(x, n, w, offset=(0, 0), dim=1):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    numel = stride[2]
    stride[1] = (seq_len + 1) * numel + stride[3]
    stride[2] = (1 if dim == 1 else seq_len) * numel
    return x.as_strided(size=(x.shape[0], n, w, w+1, *list(x.shape[4:])),
                            stride=stride,
                            storage_offset=(offset[0] * seq_len + offset[1]) * numel)

def stripe_version2_add(x, y, n, w, offset=(0, 0), dim=1):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    numel = stride[2]
    stride[1] = (seq_len + 1) * numel + stride[3]
    stride[2] = (1 if dim == 1 else seq_len) * numel

    x.as_strided(size=(x.shape[0], n, w, w+1, *list(x.shape[4:])),
                            stride=stride,
                            storage_offset=(offset[0] * seq_len + offset[1]) * numel).add_(y)


def stripe_version2_left(x, n, w):
    x = x.contiguous()
    stride = list(x.stride())
    # numel = stride[2]
    new_stride = list(x.stride())
    new_stride[1] = stride[1] + stride[2] + stride[3]
    new_stride[2] = stride[1]
    return x.as_strided(size=(x.shape[0], n-1, w, w, *list(x.shape[4:])),
                            stride=new_stride,
                            storage_offset= stride[1] +  (w+1) * stride[2] + stride[3])

# s_need_dad, n, w,
def stripe_copy_left(x, y, n, w):
    x = x.contiguous()
    stride = list(x.stride())
    # numel = stride[2]
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2] + stride[3])
    new_stride.append(stride[1])
    new_stride.extend(x.shape[4:])
    x.as_strided(size=(x.shape[0], n-1, w, *list(x.shape[4:])),
                            stride=new_stride,
                            storage_offset= stride[1] +  (w+1) * stride[2] + stride[3]).copy_(y)

def stripe_version2_right(x, n, w):
    x = x.contiguous()
    stride = list(x.stride())
    # numel = stride[2]
    new_stride = list(x.stride())
    new_stride[1] = stride[1] + stride[2] + stride[3]
    new_stride[2] = stride[2]
    return x.as_strided(size=(x.shape[0], n-1, w, w, *list(x.shape[4:])),
                            stride=new_stride,
                            storage_offset=   stride[2])

def stripe_copy_right(x, y, n, w):
    x = x.contiguous()
    stride = list(x.stride())
    # numel = stride[2]
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2] + stride[3])
    new_stride.append(stride[1])
    new_stride.extend(x.shape[4:])
    x.as_strided(size=(x.shape[0], n-1, w,  *list(x.shape[4:])),
                            stride=new_stride,
                            storage_offset= stride[2]).copy_(y)


def stripe_version_nt_nt(x, n, w, offset=(0, 0), dim=1):
    x, seq_len = x.contiguous(), x.size(2)

    stride = list(x.stride())
    numel = stride[2]
    stride[1] = (seq_len + 1) * numel + stride[3]
    stride[2] = (1 if dim == 1 else seq_len) * numel
    origin_stride = x.stride()
    return x.as_strided(size=(x.shape[0], n, w-2, w-1, *list(x.shape[4:])),
                            stride=stride,
                            storage_offset=(offset[0] * seq_len + offset[1]) * numel + \
                                          2 * (1-dim)*(origin_stride[3])
                                           + (1 - dim) * origin_stride[1] +  dim * origin_stride[2])


def stripe_version_nt_t(x, n, w):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    numel = stride[2]
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1]+stride[2]+stride[3])
    new_stride.extend(stride[3:])
    return x.as_strided(size=(x.shape[0], n, w, *list(x.shape[4:])),
                            stride=new_stride,
                            storage_offset=  w * numel)

def stripe_version_t_nt(x, n, w):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1]+stride[2]+stride[3])
    new_stride.extend(stride[3:])
    return x.as_strided(size=(x.shape[0], n, w, *list(x.shape[4:])),
                            stride=new_stride,
                            storage_offset= stride[1] + (w+1) * stride[2] + stride[3])








# used in lexicalized-pcfg.
def stripe_version4(x, n, w, offset=(0, 0), dim=1):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    numel = stride[2]
    stride[1] = (seq_len + 1) * numel + stride[3] + stride[4]
    stride[2] = (1 if dim == 1 else seq_len) * numel
    return x.as_strided(size=(x.shape[0], n, w, w+1, w+1, *list(x.shape[5:])),
                            stride=stride,
                            storage_offset=(offset[0] * seq_len + offset[1]) * numel)

# used for arc_indictor; calculating the marginal arc probabilities.
def stripe_arc_indicator(x, n, w):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    numel = stride[2]
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1]+stride[2])
    new_stride.append(stride[1])
    new_stride.append(stride[2])
    return x.as_strided(size=(x.shape[0], n, w, w),
                            stride=new_stride,
                            storage_offset=0)



#for lexicalized_pcfg.
def diagonal_copy_v2(x, y, w, lower=False):
    x, seq_len = x.contiguous(), x.size(1)
    stride, numel = list(x.stride()), x[:, 0, 0].numel()
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2] + stride[3])
    new_stride.append(stride[3])
    new_stride.extend(stride[4:])
    x.as_strided(size=(x.shape[0], seq_len - w, w, *list(x.shape[4:])),
                 stride=new_stride,
                 storage_offset= w * stride[2] if not lower else w * stride[1]
                 ).copy_(y)

def diagonal_copy_v2_for_split_point(x, y, w, lower=False):
    x, seq_len = x.contiguous(), x.size(1)
    stride, numel = list(x.stride()), x[:, 0, 0].numel()
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2] + stride[3] + stride[4])
    new_stride.append(stride[3])
    new_stride.extend(stride[4:])
    x.as_strided(size=(x.shape[0], seq_len - w, w, w, *list(x.shape[5:])),
                 stride=new_stride,
                 storage_offset= w * stride[2] if not lower else w * stride[1]
                 ).copy_(y)



def diagonal_copy_v2_add(x, y, w):
    x, seq_len = x.contiguous(), x.size(1)
    stride, numel = list(x.stride()), x[:, 0, 0].numel()
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2] + stride[3])
    new_stride.append(stride[3])
    if len(x.shape) > 4:
        new_stride.extend(stride[4:])
        x.as_strided(size=(x.shape[0], seq_len - w, w, *list(x.shape[4:])),
                     stride=new_stride,
                     storage_offset= w * stride[2]
                     ).add_(y)
    else:
        new_stride.append(stride[3])
        x.as_strided(size=(x.shape[0], seq_len - w, w, *list(x.shape[3:])),
                     stride=new_stride,
                     storage_offset= w * stride[2]
                     ).add_(y)


def diagonal2(x,  w):
    x, seq_len = x.contiguous(), x.size(1)
    stride, numel = list(x.stride()), x[:, 0, 0].numel()
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2] + stride[3])
    new_stride.append(stride[3])
    if len(x.shape) > 4:
        new_stride.extend(stride[4:])
        return x.as_strided(size=(x.shape[0], seq_len - w, w, *list(x.shape[4:])),
                     stride=new_stride,
                     storage_offset= w * stride[2]
                     )
    else:
        new_stride.append(stride[3])
        return x.as_strided(size=(x.shape[0], seq_len - w, w, *list(x.shape[3:])),
                     stride=new_stride,
                     storage_offset= w * stride[2]
                     )


#for lexicalized_pcfg.
def diagonal_copy_logadd_v2(x, y, w):
    x, seq_len = x.contiguous(), x.size(1)
    stride, numel = list(x.stride()), x[:, 0, 0].numel()
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2] + stride[3])
    new_stride.append(stride[3])
    if len(x.shape) > 4:
        new_stride.extend(stride[4:])
        tmp = x.as_strided(size=(x.shape[0], seq_len - w, w, *list(x.shape[4:])),
                     stride=new_stride,
                     storage_offset= w * stride[2]
                     )
        x.as_strided(size=(x.shape[0], seq_len - w, w, *list(x.shape[4:])),
                     stride=new_stride,
                     storage_offset=w * stride[2]
                     ).copy_(torch.logaddexp(tmp, y))
    else:
        raise NotImplementedError
        new_stride.append(stride[3])
        x.as_strided(size=(x.shape[0], seq_len - w, w, *list(x.shape[3:])),
                     stride=new_stride,
                     storage_offset= w * stride[2]
                     ).copy_(y)


def diagonal_v2(x, w, lower=False):
    x, seq_len = x.contiguous(), x.size(1)
    stride, numel = list(x.stride()), x[:, 0, 0].numel()
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2] + stride[3])
    new_stride.append(stride[3])
    new_stride.extend(stride[4:])
    return x.as_strided(size=(x.shape[0], seq_len - w, w, *list(x.shape[4:])),
                     stride=new_stride,
                     storage_offset= w * stride[2] if not lower else w*stride[1]
                     )
# def stripe()




def diagonal_copy_v4(x, y, nth_diagonal, total_num,  width, start_offset=0, head_offset=0, head_moving=0):
    x, seq_len = x.contiguous(), x.size(1)
    stride, numel = list(x.stride()), x[:, 0, 0].numel()
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2] + stride[3] + head_moving * stride[4])
    new_stride.append(stride[3])
    new_stride.extend(stride[4:])
    size = (x.shape[0], total_num, width, *list(y.shape[3:]))
    x.as_strided(size=size,
                 stride=new_stride,
                 storage_offset= start_offset * stride[1] + (start_offset + nth_diagonal) * (stride[2])   + (head_offset) * stride[4]
                 ).copy_(y)

def diagonal_copy_v3(x, y, nth_diagonal, total_num,  width, start_offset=0, head_offset=0, head_moving=1):
    x, seq_len = x.contiguous(), x.size(1)
    stride, numel = list(x.stride()), x[:, 0, 0].numel()
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2] + head_moving*stride[3])
    new_stride.append(stride[3])

    if len(x.shape) > 4:
        new_stride.extend(stride[4:])
        size = (x.shape[0], total_num, width, *list(x.shape[4:]))
        x.as_strided(size=size,
                     stride=new_stride,
                     storage_offset= start_offset * stride[1] + (start_offset + nth_diagonal) * stride[2] + (head_offset) * stride[3]
                     ).copy_(y)
    else:
        raise NotImplemented

def diagonal_copy_(x, y, w):
    # size of x: (batch, N, N, nt)
    # size of y: (batch, N, nt)
    # the function aims to copy y to the diagonal of x (dim1 and dim2) without any copy of tensor.
    x, seq_len = x.contiguous(), x.size(1)
    stride, numel = list(x.stride()), x[:, 0, 0].numel()
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2])
    if len(x.shape) > 3:
        new_stride.extend(stride[3:])
        x.as_strided(size=(x.shape[0], seq_len - w,  *list(x.shape[3:])),
                     stride=new_stride,
                     storage_offset= w * stride[2]
                     ).copy_(y)
    else:
        x.as_strided(size=(x.shape[0], seq_len - w),
                     stride=new_stride,
                     storage_offset=w * stride[2]
                     ).copy_(y)




def diagonal(x, w):
    x, seq_len = x.contiguous(), x.size(1)
    stride, numel = list(x.stride()), x[:, 0, 0].numel()
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2])
    if len(x.shape) > 3:
        new_stride.extend(stride[3:])
        return x.as_strided(size=(x.shape[0], seq_len - w, *list(x.shape[3:])),
                            stride=new_stride,
                            storage_offset=w * stride[2]
                            )
    else:
        return x.as_strided(size=(x.shape[0], seq_len - w),
                            stride=new_stride,
                            storage_offset=w * stride[2]
                            )






def stripe_compose(x, n, w):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    numel = stride[2]
    new_stride = []
    new_stride.append(stride[0])
    # new_stride.append(stride)
    stride[1] = (seq_len + 1) * numel + stride[3]
    new_stride.append(stride[1])
    new_stride.append(stride[2])
    return x.as_strided(size=(x.shape[0], n, w-1),
                            stride=new_stride,
                            storage_offset= (1) * numel + w * stride[3])
