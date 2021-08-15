# import torch.nn as nn
#
#
# class BiAAttention(nn.Module):
#     '''
#     Bi-Affine attention layer.
#     '''
#
#     def __init__(self, hparams):
#         super(BiAAttention, self).__init__()
#         self.hparams = hparams
#
#         self.dep_weight = nn.Parameter(torch_t.FloatTensor(hparams.d_biaffine + 1, hparams.d_biaffine + 1))
#         nn.init.xavier_uniform_(self.dep_weight)
#
#     def forward(self, input_d, input_e, input_s = None):
#
#         score = torch.matmul(torch.cat(
#             [input_d, torch_t.FloatTensor(input_d.size(0), 1).fill_(1).requires_grad_(False)],
#             dim=1), self.dep_weight)
#         score1 = torch.matmul(score, torch.transpose(torch.cat(
#             [input_e, torch_t.FloatTensor(input_e.size(0), 1).fill_(1).requires_grad_(False)],
#             dim=1), 0, 1))
#
#         return score1
#
# class Dep_score(nn.Module):
#     def __init__(self, hparams, num_labels):
#         super(Dep_score, self).__init__()
#
#         self.dropout_out = nn.Dropout2d(p=0.33)
#         self.hparams = hparams
#         out_dim = hparams.d_biaffine#d_biaffine
#         self.arc_h = nn.Linear(hparams.d_model, hparams.d_biaffine)
#         self.arc_c = nn.Linear(hparams.d_model, hparams.d_biaffine)
#
#         self.attention = BiAAttention(hparams)
#
#         self.type_h = nn.Linear(hparams.d_model, hparams.d_label_hidden)
#         self.type_c = nn.Linear(hparams.d_model, hparams.d_label_hidden)
#         self.bilinear = BiLinear(hparams.d_label_hidden, hparams.d_label_hidden, num_labels)
#
#     def forward(self, outputs, outpute):
#         # output from rnn [batch, length, hidden_size]
#
#         # apply dropout for output
#         # [batch, length, hidden_size] --> [batch, hidden_size, length] --> [batch, length, hidden_size]
#         outpute = self.dropout_out(outpute.transpose(1, 0)).transpose(1, 0)
#         outputs = self.dropout_out(outputs.transpose(1, 0)).transpose(1, 0)
#
#         # output size [batch, length, arc_space]
#         arc_h = nn.functional.relu(self.arc_h(outputs))
#         arc_c = nn.functional.relu(self.arc_c(outpute))
#
#         # output size [batch, length, type_space]
#         type_h = nn.functional.relu(self.type_h(outputs))
#         type_c = nn.functional.relu(self.type_c(outpute))
#
#         # apply dropout
#         # [batch, length, dim] --> [batch, 2 * length, dim]
#         arc = torch.cat([arc_h, arc_c], dim=0)
#         type = torch.cat([type_h, type_c], dim=0)
#
#         arc = self.dropout_out(arc.transpose(1, 0)).transpose(1, 0)
#         arc_h, arc_c = arc.chunk(2, 0)
#
#         type = self.dropout_out(type.transpose(1, 0)).transpose(1, 0)
#         type_h, type_c = type.chunk(2, 0)
#         type_h = type_h.contiguous()
#         type_c = type_c.contiguous()
#
#         out_arc = self.attention(arc_h, arc_c)
#         out_type = self.bilinear(type_h, type_c)
#
#         return out_arc, out_type
