# coding=utf-8
import torch
import torch.nn as nn
from operator import itemgetter
from torch.autograd import Variable
import torch.nn.functional as F
from collections import OrderedDict
import tools
import LoadFile as lf
import random


class Diagmat(nn.Module):
    def __init__(self, n):
        super(Diagmat, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(1, n).fill_(1))
        self.register_parameter("Diag", self.weight)

    def forward(self, cuda, input_sequence1, input_sequence2):
        res = input_sequence1.mul(self.weight.expand_as(input_sequence1)).mul(input_sequence2).sum(dim=1)
        return res.unsqueeze(dim=1)


class BiGRUAttention_word(nn.Module):
    def __init__(self, in_size, hidden_size, dropout=0):
        super(BiGRUAttention_word, self).__init__()
        self.register_buffer("mask", torch.FloatTensor())

        vec_size = hidden_size * 2

        self.gru = nn.GRU(input_size=in_size, hidden_size=hidden_size, num_layers=1, bias=True, batch_first=True,
                          dropout=dropout, bidirectional=True)
        self.lin = nn.Linear(hidden_size * 2, vec_size)
        self.attention = nn.Linear(vec_size, 1, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, packed_batch):
        rnn_v, _ = self.gru(packed_batch)
        emb_v, len_seq = torch.nn.utils.rnn.pad_packed_sequence(rnn_v) 
        emb_h = self.tanh(self.lin(emb_v.view(emb_v.size(0) * emb_v.size(1), -1)))
        att = self.attention(emb_h).view(emb_v.size(0), emb_v.size(1)).transpose(0, 1)
        att_soft_max_mask = self._masked_softmax(att, self._list_to_bytemask(list(len_seq))).transpose(0,1) 
        emb_v_att = att_soft_max_mask.unsqueeze(2).expand_as(emb_v) * emb_v
        return emb_v_att.sum(0, True).squeeze(0)

    def _list_to_bytemask(self, l):
        mask = self._buffers['mask'].resize_(len(l), l[0]).fill_(1)
        for i, j in enumerate(l):
            if j != l[0]:
                mask[i, j:l[0]] = 0
        return mask

    def _masked_softmax(self, mat, mask):
        exp = torch.exp(mat) * Variable(mask, requires_grad=False)
        sum_exp = exp.sum(1, True) + 0.0001
        return exp / sum_exp.expand_as(exp)

    def forword_test(self, packed_batch):
        rnn_v, _ = self.gru(packed_batch)
        emb_v, len_seq = torch.nn.utils.rnn.pad_packed_sequence(rnn_v)
        emb_h = self.tanh(self.lin(emb_v.view(emb_v.size(0) * emb_v.size(1), -1)))
        att = self.attention(emb_h).view(emb_v.size(0), emb_v.size(1)).transpose(0, 1)
        att_soft_max_mask = self._masked_softmax(att, self._list_to_bytemask(list(len_seq))).transpose(0,
                                                                                                       1)
        emb_v_att = att_soft_max_mask.unsqueeze(2).expand_as(emb_v) * emb_v
        return emb_v_att.sum(0, True).squeeze(0), att
class BiGRUAttention_sent(nn.Module):
    def __init__(self, in_size, hidden_size, dropout=0):
        super(BiGRUAttention_sent, self).__init__()
        self.register_buffer("mask", torch.FloatTensor())

        vec_size = hidden_size * 2

        self.gru = nn.GRU(input_size=in_size, hidden_size=hidden_size, num_layers=1, bias=True, batch_first=True,
                          dropout=dropout, bidirectional=True)
        self.lin = nn.Linear(hidden_size * 2*2, vec_size*2)
        self.attention = nn.Linear(vec_size*2, 1, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, packed_batch,q_vs):
        rnn_v, _ = self.gru(packed_batch)
        emb_v, len_seq = torch.nn.utils.rnn.pad_packed_sequence(rnn_v)
        q_vs = q_vs.unsqueeze(dim=0)

        q_vs = q_vs.expand_as(emb_v)
        mat = torch.cat((emb_v, q_vs), dim=2)
        emb_h = self.tanh(self.lin(mat.view(emb_v.size(0) * emb_v.size(1), -1)))
        att = self.attention(emb_h).view(emb_v.size(0), emb_v.size(1)).transpose(0, 1)
        att_soft_max_mask = self._masked_softmax(att, self._list_to_bytemask(list(len_seq))).transpose(0,
                                                                                                       1)
        emb_v_att = att_soft_max_mask.unsqueeze(2).expand_as(emb_v) * emb_v
        return emb_v_att.sum(0, True).squeeze(0)

    def _list_to_bytemask(self, l):
        mask = self._buffers['mask'].resize_(len(l), l[0]).fill_(1)
        for i, j in enumerate(l):
            if j != l[0]:
                mask[i, j:l[0]] = 0
        return mask

    def _masked_softmax(self, mat, mask):
        exp = torch.exp(mat) * Variable(mask, requires_grad=False) 
        sum_exp = exp.sum(1, True) + 0.0001

        return exp / sum_exp.expand_as(exp)

    def forword_test(self, packed_batch):
        rnn_v, _ = self.gru(packed_batch)
        emb_v, len_seq = torch.nn.utils.rnn.pad_packed_sequence(rnn_v)
        emb_h = self.tanh(self.lin(emb_v.view(emb_v.size(0) * emb_v.size(1), -1)))
        att = self.attention(emb_h).view(emb_v.size(0), emb_v.size(1)).transpose(0, 1)
        att_soft_max_mask = self._masked_softmax(att, self._list_to_bytemask(list(len_seq))).transpose(0,1) 
        emb_v_att = att_soft_max_mask.unsqueeze(2).expand_as(emb_v) * emb_v
        return emb_v_att.sum(0, True).squeeze(0), att

class HierarchicalQANet(nn.Module):
    def __init__(self, len_word_list, emb_s=300, hid_s=150):
        super(HierarchicalQANet, self).__init__()
        self.register_buffer("reviews", torch.Tensor())
        self.register_buffer("topics", torch.Tensor())
        self.register_buffer("users", torch.Tensor())

        self.emb_size = emb_s
        self.vec_size = hid_s * 2
        self.embed = nn.Embedding(len_word_list, emb_s, padding_idx=0)

        self.word = BiGRUAttention_word(self.emb_size, hid_s)
        self.sent = BiGRUAttention_sent(self.vec_size, hid_s)

        self.topic_attention_att2 = nn.Linear(hid_s * 2, hid_s * 2, bias=False)
        self.user_attention_att2 = nn.Linear(hid_s * 2, hid_s * 2, bias=False)

        self.topic_attention_att3 = nn.Linear(hid_s * 2*2, hid_s * 2, bias=False)
        self.user_attention_att3 = nn.Linear(hid_s * 2*2, hid_s * 2, bias=False)
        self.topic_vc_att3=nn.Linear(hid_s * 2,1,bias=False)
        self.user_vc_att3=nn.Linear(hid_s * 2,1,bias=False)

        self.tanh_topic = nn.Tanh()
        self.tanh_user = nn.Tanh()

        self.final_score_2 = nn.Linear(2, 1)
        self.final_score_3 = nn.Linear(3, 1)
        self.final_score_4 = nn.Linear(4, 1)
        self.final_score_5 = nn.Linear(5, 1)
        self.final_score_6 = nn.Linear(6, 1)

    def set_emb_tensor(self, emb_tensor):
        self.emb_size = emb_tensor.size(-1)
        self.embed.weight.data = emb_tensor

    def _reorder_sent_new(self, cuda, sent_embs, info_qtu, info_sent, topic_info, user_info):

        sort_sent = sorted([(qn, sn, emb_index) for emb_index, (qn, sn) in enumerate(info_sent)], key=itemgetter(0, 1))
        emb_real_order = [(inf[0], inf[1], sent_embs[inf[2]]) for i, inf in enumerate(sort_sent)]

        topic_count = len(topic_info)
        user_count = len(user_info)
        qa_count = len(
            set([qn for qn, sn, emb_index in sort_sent])) - topic_count - user_count
        qa_question_count = 0
        for (qn, sn, _) in emb_real_order:
            if qn >= qa_count:
                break
            qa_question_count += 1
        emb_real_order = emb_real_order[:qa_question_count]

        sort_all_vec = sorted([[lq, qn, sn, emb_index] for emb_index, (lq, qn, sn) in enumerate(info_qtu)],
                              key=itemgetter(0, 1, 2))
        sort_qa = []
        sort_topic = []
        sort_user = []
        q_index=[]
        for sorted_sent_index, (lq, qn, sn, emb_index) in enumerate(sort_all_vec):
            if qn < qa_count:
                if sn==0:
                    q_index.append(emb_index)
                sort_qa.append([lq, qn, sn, emb_index])
            elif qn >= qa_count and qn < qa_count + topic_count:
                sort_topic.append([lq, qn, sn, emb_index])
            else:
                sort_user.append([lq, qn, sn, emb_index])


        qa_dic = OrderedDict()
        for lq, qn, sn, emb_index in sort_qa:
            if qn not in qa_dic:
                qa_dic[qn] = [emb_index]
            else:
                qa_dic[qn].append(emb_index)

        user_dic = OrderedDict()
        for lq, qn, sn, emb_index in sort_user:
            if qn not in user_dic:
                user_dic[qn] = [emb_index]
            else:
                user_dic[qn].append(emb_index)
        topic_dic = OrderedDict()
        for lq, qn, sn, emb_index in sort_topic:
            if qn not in topic_dic:
                topic_dic[qn] = [emb_index]
            else:
                topic_dic[qn].append(emb_index)

        qa_index = list(reversed(qa_dic))
        qa_vs = Variable(
            self._buffers["reviews"].resize_(len(qa_dic), len(qa_dic[qa_index[0]]), sent_embs.size(1)).fill_(0),
            requires_grad=False)
        q_vs=torch.zeros(len(qa_index),sent_embs.size(1))
        qa_lens = []
        real_order = []

        for i, qn in enumerate(qa_index):
            qa_vs[i, 0:len(qa_dic[qa_index[i]]), :] = sent_embs[qa_dic[qa_index[i]], :]
            q_vs[i,:]=sent_embs[q_index[i], :]
            qa_lens.append(len(qa_dic[qa_index[i]]))
            real_order.append(qn)

        topic_index = list(topic_dic)
        if len(topic_index) != 0:
            topic_vs = Variable(
                self._buffers["topics"].resize_(len(topic_dic), len(topic_dic[topic_index[-1]]),sent_embs.size(1)).fill_(0),
                requires_grad=False)
            for i, qn in enumerate(topic_index):
                topic_vs[qn - qa_count, 0:len(topic_dic[topic_index[i]]), :] = sent_embs[topic_dic[topic_index[i]], :]
        else:
            topic_vs = None
        user_index = list(user_dic)
        if len(user_index) != 0:
            user_vs = Variable(
                self._buffers["users"].resize_(len(user_dic), len(user_dic[user_index[-1]]), sent_embs.size(1)).fill_(
                    0),
                requires_grad=False)
            for i, qn in enumerate(user_index):
                user_vs[(qn - topic_count - qa_count), 0:len(user_dic[user_index[i]]), :] = sent_embs[
                                                                                            user_dic[user_index[i]], :]
        else:
            user_vs = None
        return qa_vs, qa_lens,q_vs, real_order, tuple(emb_real_order), topic_vs, user_vs

    def build_final_layer_input(self, cuda, qa_vec_list, sent_vec_list, index_len, ent_vec, mention_info, ent_id,
                                topic_vecs_list, user_vecs_list, last_res):
        sent_index = 0
        final_info = []
        no_candi_count = 0
        all_count = 0
        mention_index = 0

        all_candi_count = sum([len(c_list) for _, _, mention_list in mention_info for _, _, c_list in mention_list])

        qa_sim_input = torch.FloatTensor(all_candi_count, qa_vec_list.size()[-1])
        sent_sim_input = torch.FloatTensor(all_candi_count, qa_vec_list.size()[-1])

        candi_popu = torch.FloatTensor(all_candi_count, 1)
        candi_sim_input = torch.FloatTensor(all_candi_count, qa_vec_list.size()[-1])
        if cuda:
            qa_sim_input = qa_sim_input.cuda()
            sent_sim_input = sent_sim_input.cuda()
            candi_popu = candi_popu.cuda()
            candi_sim_input = candi_sim_input.cuda()
        if type(topic_vecs_list) != type(None):
            topic_sim_input = torch.FloatTensor(all_candi_count, qa_vec_list.size()[-1])
            if cuda:
                topic_sim_input = topic_sim_input.cuda()
        if type(user_vecs_list) != type(None):
            user_sim_input = torch.FloatTensor(all_candi_count, qa_vec_list.size()[-1])
            if cuda:
                user_sim_input = user_sim_input.cuda()
        if last_res:
            last_vec_list = self.buildsumvec(last_res, [i for i, l in index_len])
            last_res_sim_input = torch.FloatTensor(all_candi_count, qa_vec_list.size()[-1])
            if cuda:
                last_res_sim_input = last_res_sim_input.cuda()

        all_candi_index = 0
        for i, (q_index, q_len) in enumerate(index_len):
            qa_vector = qa_vec_list[i] 
            for s_index in range(q_len):
                s_vector = sent_vec_list[sent_index][2]
                if type(user_vecs_list) != type(None):
                    u_vector = user_vecs_list[sent_index]
                sent_index += 1
                mentions = mention_info[mention_index][2]  # [(m,e)]
                mention_index += 1
                if mentions == None:
                    continue
                for m_index, mae in enumerate(mentions):
                    all_count += 1
                    m = mae[0]
                    e = mae[1]
                    if e in ent_id:
                        e_id = ent_id[e]
                    else:
                        print(e)
                    candi_info = mae[2]
                    for c, c_id, popu in candi_info:
                        c_vector = lf.get_entity_vec(ent_vec, c_id)
                        if type(c_vector) == type(None):
                            print("candidate entity no id ".format(c_id))
                            exit(1)
                        if c == e:
                            final_info.append([q_index, s_index, m_index, c_id, c, True])
                        else:
                            final_info.append([q_index, s_index, m_index, c_id, c, False])
                        candi_popu[all_candi_index] = -1 * torch.log(popu)
                        qa_sim_input[all_candi_index] = qa_vector
                        sent_sim_input[all_candi_index] = s_vector
                        candi_sim_input[all_candi_index] = c_vector
                        if type(topic_vecs_list) != type(None):
                            topic_sim_input[all_candi_index] = topic_vecs_list[i]
                        if type(user_vecs_list) != type(None):
                            user_sim_input[all_candi_index] = u_vector
                        if last_res:
                            last_res_sim_input[all_candi_index] = last_vec_list[i][s_index]
                        all_candi_index += 1
        qa_sim = self.qa_sim_layer(cuda, qa_sim_input, candi_sim_input)
        sent_sim = self.sent_sim_layer(cuda, sent_sim_input, candi_sim_input)
        final_input = torch.cat((qa_sim, sent_sim), dim=1)

        if type(topic_vecs_list) != type(None):
            topic_sim = self.topic_sim_layer(cuda, topic_sim_input, candi_sim_input)
            final_input = torch.cat((final_input, topic_sim), dim=1)
        if type(user_vecs_list) != type(None):
            user_sim = self.user_sim_layer(cuda, user_sim_input, candi_sim_input)
            final_input = torch.cat((final_input, user_sim), dim=1)
        if last_res:
            last_res_sim = self.last_res_sim_layer(cuda, last_res_sim_input, candi_sim_input)
            final_input = torch.cat((final_input, last_res_sim), dim=1)

        final_input = torch.cat((final_input, candi_popu), dim=1)
        return final_input, final_info, no_candi_count

    def build_final_layer_input_onecat(self, cuda, qa_vec_list, sent_vec_list, index_len, ent_vec, mention_info, ent_id,
                                    topic_vecs_list, user_vecs_list, last_res):
        if last_res:
            last_vec_list = self.buildsumvec(last_res, [i for i, l in index_len])
        sent_index = 0
        final_input = torch.FloatTensor()
        if cuda:
            final_input=final_input.cuda()
        final_info = []
        no_candi_count = 0
        all_count = 0
        mention_index = 0

        for i, (q_index, q_len) in enumerate(index_len):
            qa_vector = qa_vec_list[i]
            if type(topic_vecs_list)!=type(None):
                t_vector = topic_vecs_list[i]
            for s_index in range(q_len):
                s_vector = sent_vec_list[sent_index][2]

                if type(user_vecs_list) != type(None):
                    u_vector = user_vecs_list[sent_index]
                sent_index += 1
                mentions = mention_info[mention_index][2]
                mention_index += 1
                if mentions == None:
                    continue
                for m_index, mae in enumerate(mentions):
                    all_count += 1
                    m = mae[0]
                    e = mae[1]
                    if e in ent_id:
                        e_id = ent_id[e]
                    else:
                        print(e)
                    candi_info = mae[2]
                    for c, c_id, popu in candi_info:
                        c_vector = lf.get_entity_vec(ent_vec, c_id)
                        if type(c_vector) == type(None):
                            print("candidate entity no id ".format(c_id))
                            continue
                        if c == e:
                            final_info.append([q_index, s_index, m_index, c_id, c, True])
                        else:
                            final_info.append([q_index, s_index, m_index, c_id, c, False])
                        if type(topic_vecs_list)==type(None) and type(user_vecs_list) == type(None):
                            if last_res:
                                last_vector=last_vec_list[i]
                                final_input = torch.cat((final_input,
                                                         torch.cat((torch.cosine_similarity(qa_vector, c_vector,
                                                                                            dim=0).unsqueeze(dim=0),
                                                                    torch.cosine_similarity(s_vector, c_vector,
                                                                                            dim=0).unsqueeze(dim=0),
                                                                    torch.cosine_similarity(last_vector, c_vector,
                                                                                            dim=0).unsqueeze(dim=0),
                                                                    -1 * torch.log(popu)),
                                                                   dim=0).unsqueeze(dim=0)),
                                                        dim=0)
                            else:
                                final_input = torch.cat((final_input,
                                                     torch.cat((torch.cosine_similarity(qa_vector, c_vector,
                                                                                        dim=0).unsqueeze(dim=0),
                                                                torch.cosine_similarity(s_vector, c_vector,
                                                                                        dim=0).unsqueeze(dim=0),
                                                                -1 * torch.log(popu)),
                                                               dim=0).unsqueeze(dim=0)),
                                                    dim=0)
                        elif type(topic_vecs_list)==type(None):
                            if last_res:
                                last_vector=last_vec_list[i]
                                final_input = torch.cat((final_input,
                                                         torch.cat((torch.cosine_similarity(qa_vector, c_vector,
                                                                                            dim=0).unsqueeze(dim=0),
                                                                    torch.cosine_similarity(s_vector, c_vector,
                                                                                            dim=0).unsqueeze(dim=0),
                                                                    torch.cosine_similarity(u_vector, c_vector,
                                                                                            dim=0).unsqueeze(dim=0),
                                                                    torch.cosine_similarity(last_vector, c_vector,
                                                                                            dim=0).unsqueeze(dim=0),
                                                                    -1 * torch.log(popu)),
                                                                   dim=0).unsqueeze(dim=0)),
                                                        dim=0)
                            else:
                                final_input = torch.cat((final_input,
                                                     torch.cat((torch.cosine_similarity(qa_vector, c_vector,
                                                                                        dim=0).unsqueeze(dim=0),
                                                                torch.cosine_similarity(s_vector, c_vector,
                                                                                        dim=0).unsqueeze(dim=0),
                                                                torch.cosine_similarity(u_vector, c_vector,
                                                                                        dim=0).unsqueeze(dim=0),
                                                                -1 * torch.log(popu)),
                                                               dim=0).unsqueeze(dim=0)),
                                                    dim=0)
                        elif type(user_vecs_list) == type(None):
                            if last_res:
                                last_vector=last_vec_list[i]
                                final_input = torch.cat((final_input,
                                                         torch.cat((torch.cosine_similarity(qa_vector, c_vector,
                                                                                            dim=0).unsqueeze(dim=0),
                                                                    torch.cosine_similarity(s_vector, c_vector,
                                                                                            dim=0).unsqueeze(dim=0),
                                                                    torch.cosine_similarity(t_vector, c_vector,
                                                                                            dim=0).unsqueeze(dim=0),
                                                                    torch.cosine_similarity(last_vector, c_vector,
                                                                                            dim=0).unsqueeze(dim=0),
                                                                    -1 * torch.log(popu)),
                                                                   dim=0).unsqueeze(dim=0)),
                                                        dim=0)
                            else:
                                final_input = torch.cat((final_input,
                                                     torch.cat((torch.cosine_similarity(qa_vector, c_vector,
                                                                                        dim=0).unsqueeze(dim=0),
                                                                torch.cosine_similarity(s_vector, c_vector,
                                                                                        dim=0).unsqueeze(dim=0),
                                                                torch.cosine_similarity(t_vector, c_vector,
                                                                                        dim=0).unsqueeze(dim=0),
                                                                -1 * torch.log(popu)),
                                                               dim=0).unsqueeze(dim=0)),
                                                    dim=0)
                        else:
                            if last_res:
                                last_vector=last_vec_list[i]
                                final_input = torch.cat((final_input,
                                                         torch.cat((torch.cosine_similarity(qa_vector, c_vector,
                                                                                            dim=0).unsqueeze(dim=0),
                                                                    torch.cosine_similarity(s_vector, c_vector,
                                                                                            dim=0).unsqueeze(dim=0),
                                                                    torch.cosine_similarity(t_vector, c_vector,
                                                                                            dim=0).unsqueeze(dim=0),
                                                                    torch.cosine_similarity(u_vector, c_vector,
                                                                                            dim=0).unsqueeze(dim=0),
                                                                    torch.cosine_similarity(last_vector, c_vector,
                                                                                            dim=0).unsqueeze(dim=0),
                                                                    -1 * torch.log(popu)),
                                                                   dim=0).unsqueeze(dim=0)),
                                                        dim=0)
                            else:
                                final_input = torch.cat((final_input,
                                                     torch.cat((torch.cosine_similarity(qa_vector, c_vector,
                                                                                        dim=0).unsqueeze(dim=0),
                                                                torch.cosine_similarity(s_vector, c_vector,
                                                                                        dim=0).unsqueeze(dim=0),
                                                                torch.cosine_similarity(t_vector, c_vector,
                                                                                        dim=0).unsqueeze(dim=0),
                                                                torch.cosine_similarity(u_vector, c_vector,
                                                                                        dim=0).unsqueeze(dim=0),
                                                                -1 * torch.log(popu)),
                                                               dim=0).unsqueeze(dim=0)),
                                                    dim=0)
        return final_input, final_info, no_candi_count

    def build_final_layer_input_onecat_onlyqa(self, cuda, qa_vec_list, sent_vec_list, index_len, ent_vec, mention_info, ent_id,
                                    topic_vecs_list, user_vecs_list, last_res):
        if last_res:
            last_vec_list = self.buildsumvec(last_res, [i for i, l in index_len])
        sent_index = 0
        final_input = torch.FloatTensor()
        if cuda:
            final_input=final_input.cuda()
        final_info = []
        no_candi_count = 0
        all_count = 0
        mention_index = 0

        for i, (q_index, q_len) in enumerate(index_len):
            qa_vector = qa_vec_list[i]
            if type(topic_vecs_list)!=type(None):
                t_vector = topic_vecs_list[i]
            for s_index in range(q_len):
                s_vector = sent_vec_list[sent_index][2]

                if type(user_vecs_list) != type(None):
                    u_vector = user_vecs_list[sent_index]
                sent_index += 1
                mentions = mention_info[mention_index][2]
                mention_index += 1
                if mentions == None:
                    continue
                for m_index, mae in enumerate(mentions):
                    all_count += 1
                    m = mae[0]
                    e = mae[1]
                    if e in ent_id:
                        e_id = ent_id[e]
                    else:
                        print(e)
                    candi_info = mae[2]
                    for c, c_id, popu in candi_info:
                        c_vector = lf.get_entity_vec(ent_vec, c_id)
                        if type(c_vector) == type(None):
                            print("candidate entity no id ".format(c_id))
                            continue
                        if c == e:
                            final_info.append([q_index, s_index, m_index, c_id, c, True])
                        else:
                            final_info.append([q_index, s_index, m_index, c_id, c, False])
                        if type(topic_vecs_list)==type(None) and type(user_vecs_list) == type(None):
                            if last_res:
                                last_vector=last_vec_list[i]
                                final_input = torch.cat((final_input,
                                                         torch.cat((torch.cosine_similarity(qa_vector, c_vector,
                                                                                            dim=0).unsqueeze(dim=0),
                                                                    torch.cosine_similarity(last_vector, c_vector,
                                                                                            dim=0).unsqueeze(dim=0),
                                                                    -1 * torch.log(popu)),
                                                                   dim=0).unsqueeze(dim=0)),
                                                        dim=0)
                            else:
                                final_input = torch.cat((final_input,
                                                     torch.cat((torch.cosine_similarity(qa_vector, c_vector,
                                                                                        dim=0).unsqueeze(dim=0),
                                                                -1 * torch.log(popu)),
                                                               dim=0).unsqueeze(dim=0)),
                                                    dim=0)
                        elif type(topic_vecs_list)==type(None):
                            if last_res:
                                last_vector=last_vec_list[i]
                                final_input = torch.cat((final_input,
                                                         torch.cat((torch.cosine_similarity(qa_vector, c_vector,
                                                                                            dim=0).unsqueeze(dim=0),
                                                                    torch.cosine_similarity(u_vector, c_vector,
                                                                                            dim=0).unsqueeze(dim=0),
                                                                    torch.cosine_similarity(last_vector, c_vector,
                                                                                            dim=0).unsqueeze(dim=0),
                                                                    -1 * torch.log(popu)),
                                                                   dim=0).unsqueeze(dim=0)),
                                                        dim=0)
                            else:
                                final_input = torch.cat((final_input,
                                                     torch.cat((torch.cosine_similarity(qa_vector, c_vector,
                                                                                        dim=0).unsqueeze(dim=0),
                                                                torch.cosine_similarity(u_vector, c_vector,
                                                                                        dim=0).unsqueeze(dim=0),
                                                                -1 * torch.log(popu)),
                                                               dim=0).unsqueeze(dim=0)),
                                                    dim=0)
                        elif type(user_vecs_list) == type(None):
                            if last_res:
                                last_vector=last_vec_list[i]
                                final_input = torch.cat((final_input,
                                                         torch.cat((torch.cosine_similarity(qa_vector, c_vector,
                                                                                            dim=0).unsqueeze(dim=0),
                                                                    torch.cosine_similarity(t_vector, c_vector,
                                                                                            dim=0).unsqueeze(dim=0),
                                                                    torch.cosine_similarity(last_vector, c_vector,
                                                                                            dim=0).unsqueeze(dim=0),
                                                                    -1 * torch.log(popu)),
                                                                   dim=0).unsqueeze(dim=0)),
                                                        dim=0)
                            else:
                                final_input = torch.cat((final_input,
                                                     torch.cat((torch.cosine_similarity(qa_vector, c_vector,
                                                                                        dim=0).unsqueeze(dim=0),
                                                                torch.cosine_similarity(t_vector, c_vector,
                                                                                        dim=0).unsqueeze(dim=0),
                                                                -1 * torch.log(popu)),
                                                               dim=0).unsqueeze(dim=0)),
                                                    dim=0)
                        else:
                            if last_res:
                                last_vector=last_vec_list[i]
                                final_input = torch.cat((final_input,
                                                         torch.cat((torch.cosine_similarity(qa_vector, c_vector,
                                                                                            dim=0).unsqueeze(dim=0),
                                                                    torch.cosine_similarity(t_vector, c_vector,
                                                                                            dim=0).unsqueeze(dim=0),
                                                                    torch.cosine_similarity(u_vector, c_vector,
                                                                                            dim=0).unsqueeze(dim=0),
                                                                    torch.cosine_similarity(last_vector, c_vector,
                                                                                            dim=0).unsqueeze(dim=0),
                                                                    -1 * torch.log(popu)),
                                                                   dim=0).unsqueeze(dim=0)),
                                                        dim=0)
                            else:
                                final_input = torch.cat((final_input,
                                                     torch.cat((torch.cosine_similarity(qa_vector, c_vector,
                                                                                        dim=0).unsqueeze(dim=0),
                                                                torch.cosine_similarity(t_vector, c_vector,
                                                                                        dim=0).unsqueeze(dim=0),
                                                                torch.cosine_similarity(u_vector, c_vector,
                                                                                        dim=0).unsqueeze(dim=0),
                                                                -1 * torch.log(popu)),
                                                               dim=0).unsqueeze(dim=0)),
                                                    dim=0)
        return final_input, final_info, no_candi_count

    def forward(self, cuda, batch_sentence, info, index_len, ent_vec, ent_id, topic_info, user_info, mention_info,
                topic,user, last_res):
        final_size = 3
        ls, lq, qn, sn = zip(*info)
        emb_init = self.embed(batch_sentence)
        emb_w = F.dropout(emb_init, training=self.training)

        packed_sents = torch.nn.utils.rnn.pack_padded_sequence(emb_w, ls, batch_first=True)

        sent_embs = self.word(packed_sents)

        qs_embs, lens, q_vs,real_order, real_sent_embs, topic_qs_vecs_mat, user_qs_vecs_mat = self._reorder_sent_new(
            cuda, sent_embs, zip(lq, qn, sn), zip(qn, sn), topic_info, user_info)

        packed_qs = torch.nn.utils.rnn.pack_padded_sequence(qs_embs, lens, batch_first=True)

        qa_embs = self.sent(packed_qs,q_vs)
        real_qa_emb = qa_embs[real_order, :]

        if topic and user:
            # print("topic + user")
            final_size += 2

            # attention 1
            # 使用att2的函数实现att1
            # topic_att_dateset = self.cat_topic_vec_att2(cuda, real_qa_emb, topic_info, topic_qs_vecs_mat)
            # topic_vecs = self.rebuild_topic_vec_top_att2(cuda, topic_qs_vecs_mat,
            #                     topic_att_dateset.view(topic_att_dateset.size()[0] * topic_att_dateset.size()[1], -1),
            #                                             topic_info, real_qa_emb)
            # user_att_dateset = self.cat_user_vec_att2(cuda, real_qa_emb, user_info, user_qs_vecs_mat)
            # user_vecs = self.rebuild_user_vec_top_att2(cuda, user_qs_vecs_mat,
            #                     user_att_dateset.view(user_att_dateset.size()[0] * user_att_dateset.size(1), -1),
            #                                            user_info, real_qa_emb)

            # attention 2
            topic_att_dateset = self.cat_topic_vec_att2(cuda, real_qa_emb, topic_info, topic_qs_vecs_mat)
            topic_att = self.tanh_topic(
                self.topic_attention_att2(
                    topic_att_dateset.view(topic_att_dateset.size()[0] * topic_att_dateset.size()[1], -1)))
            topic_vecs = self.rebuild_topic_vec_top_att2(cuda, topic_qs_vecs_mat, topic_att, topic_info, real_qa_emb)
            #
            # user_att_dateset = self.cat_user_vec_att2(cuda, real_qa_emb, user_info, user_qs_vecs_mat)
            # user_att = self.tanh_user(
            #     self.user_attention_att2(
            #         user_att_dateset.view(user_att_dateset.size()[0] * user_att_dateset.size(1), -1)))
            # user_vecs = self.rebuild_user_vec_top_att2(cuda, user_qs_vecs_mat, user_att, user_info, real_qa_emb)

            # attention 3
            # user_att_dateset = self.cat_user_vec_att3(cuda, real_qa_emb, user_info, user_qs_vecs_mat)
            # user_att_value = self.user_vc_att3(self.tanh_user(
            #     self.user_attention_att3(
            #         user_att_dateset.view(user_att_dateset.size()[0] * user_att_dateset.size(1), -1))))
            # user_vecs=self.rebuild_user_vec_top_att3(cuda, user_qs_vecs_mat, user_att_value, user_info, real_qa_emb)
            #
            # topic_att_dateset = self.cat_topic_vec_att3(cuda, real_qa_emb, topic_info, topic_qs_vecs_mat)
            # topic_att_value = self.topic_vc_att3(self.tanh_topic(
            #     self.topic_attention_att3(
            #         topic_att_dateset.view(topic_att_dateset.size()[0] * topic_att_dateset.size()[1], -1))))
            # topic_vecs=self.rebuild_topic_vec_top_att3(cuda, topic_qs_vecs_mat, topic_att_value, topic_info, real_qa_emb)

            #non-att
            # topic_att_dateset = self.cat_topic_vec_att2(cuda, real_qa_emb, topic_info, topic_qs_vecs_mat)
            # topic_vecs = self.rebuild_topic_vec_noneatt(cuda, topic_att_dateset, topic_info)
            # user_att_dateset = self.cat_user_vec_att2(cuda, real_qa_emb, user_info, user_qs_vecs_mat)
            # user_vecs = self.rebuild_user_vec_noneatt(cuda, user_att_dateset, user_info)

        elif topic:
            # print("topic")
            final_size += 1
            #attention 1
            #使用att2的函数实现att1
            # topic_att_dateset = self.cat_topic_vec_att2(cuda, real_qa_emb, topic_info, topic_qs_vecs_mat)
            # topic_vecs = self.rebuild_topic_vec_top_att2(cuda, topic_qs_vecs_mat,
            #                     topic_att_dateset.view(topic_att_dateset.size()[0] * topic_att_dateset.size()[1], -1),
            #                                             topic_info, real_qa_emb)
            #attention 2
            topic_att_dateset = self.cat_topic_vec_att2(cuda, real_qa_emb, topic_info, topic_qs_vecs_mat)
            topic_att = self.tanh_topic(
                self.topic_attention_att2(
                    topic_att_dateset.view(topic_att_dateset.size()[0] * topic_att_dateset.size()[1], -1)))
            topic_vecs = self.rebuild_topic_vec_top_att2(cuda, topic_qs_vecs_mat, topic_att, topic_info, real_qa_emb)

            #attention 3
            # topic_att_dateset = self.cat_topic_vec_att3(cuda, real_qa_emb, topic_info, topic_qs_vecs_mat)
            # topic_att_value = self.topic_vc_att3(self.tanh_topic(
            #     self.topic_attention_att3(
            #         topic_att_dateset.view(topic_att_dateset.size()[0] * topic_att_dateset.size()[1], -1))))
            # topic_vecs=self.rebuild_topic_vec_top_att3(cuda, topic_qs_vecs_mat, topic_att_value, topic_info, real_qa_emb)
            # topic_att_dateset = self.cat_topic_vec_att2(cuda, real_qa_emb, topic_info, topic_qs_vecs_mat)
            # topic_vecs = self.rebuild_topic_vec_noneatt(cuda, topic_att_dateset, topic_info)

            user_vecs = None
        elif user:
            # print("user")
            final_size += 1
            #attention 1
            # 使用att2 的函数完成att1
            # user_att_dateset = self.cat_user_vec_att2(cuda, real_qa_emb, user_info, user_qs_vecs_mat)
            # user_vecs = self.rebuild_user_vec_top_att2(cuda, user_qs_vecs_mat,
            #                     user_att_dateset.view(user_att_dateset.size()[0] * user_att_dateset.size(1), -1),
            #                                            user_info, real_qa_emb)
            #attention 2
            user_att_dateset = self.cat_user_vec_att2(cuda, real_qa_emb, user_info, user_qs_vecs_mat)
            user_att = self.tanh_user(
                self.user_attention_att2(user_att_dateset.view(user_att_dateset.size()[0] * user_att_dateset.size(1), -1)))
            user_vecs = self.rebuild_user_vec_top_att2(cuda, user_qs_vecs_mat, user_att, user_info, real_qa_emb)

            # attention 3
            # user_att_dateset = self.cat_user_vec_att3(cuda, real_qa_emb, user_info, user_qs_vecs_mat)
            # user_att_value = self.user_vc_att3(self.tanh_user(
            #     self.user_attention_att3(
            #         user_att_dateset.view(user_att_dateset.size()[0] * user_att_dateset.size(1), -1))))
            # user_vecs=self.rebuild_user_vec_top_att3(cuda, user_qs_vecs_mat, user_att_value, user_info, real_qa_emb)
            # user_att_dateset = self.cat_user_vec_att2(cuda, real_qa_emb, user_info, user_qs_vecs_mat)
            # user_vecs = self.rebuild_user_vec_noneatt(cuda, user_att_dateset, user_info)
            topic_vecs = None
        else:
            # print("None")
            topic_vecs = None
            user_vecs = None

        final_input, final_info, no_candi_count = self.build_final_layer_input_onecat(cuda, real_qa_emb, real_sent_embs,
                                                                                   index_len, ent_vec, mention_info,
                                                                                   ent_id, topic_vecs, user_vecs,
                                                                                   last_res)
        # final_size-=1#qa 与 sent 二选一的时候需要减1
        if last_res:
            final_size += 1
        if final_size == 2:
            out = self.final_score_2(final_input)
        elif final_size == 3:
            out = self.final_score_3(final_input)
        elif final_size == 4:
            out = self.final_score_4(final_input)
        elif final_size == 5:
            out = self.final_score_5(final_input)
        else:
            out = self.final_score_6(final_input)

        return out, final_info, no_candi_count

    def rebuild_topic_vec_new(self, cuda, topic_qs_vecs, att_vec_list, topic_info, qa_vecs):
        if cuda:
            att_list = Variable(torch.FloatTensor(att_vec_list.size()[0], 1).fill_(0).cuda())
        else:
            att_list = Variable(torch.FloatTensor(att_vec_list.size()[0], 1).fill_(0))
        if len(topic_info) > 0:
            qa_index = topic_info[0][0]
        else:
            return
        qa_i = 0
        att_vec_index = 0
        for t_i, t_info in enumerate(topic_info):
            if t_i > 0:
                if t_info[0] != qa_index:
                    qa_index = t_info[0]
                    qa_i += 1
            for t_q_i in range(topic_qs_vecs.size()[1]):
                ppp = att_vec_list[att_vec_index].unsqueeze(0).mm(qa_vecs[qa_i].unsqueeze(1)).squeeze(1)
                att_list[att_vec_index] = ppp
                att_vec_index += 1
        att_list_e = torch.exp(att_list).view(topic_qs_vecs.size()[0], topic_qs_vecs.size()[1], -1)
        soft_max_att_mat = torch.zeros(len(topic_info), topic_qs_vecs.size()[1]).float()
        if cuda:
            soft_max_att_mat = soft_max_att_mat.cuda()
        for i in range(len(topic_info)):
            for j in range(topic_info[i][2]):
                soft_max_att_mat[i][j] = att_list_e[i][j]
        exp_sum = soft_max_att_mat.sum(dim=1)
        soft_max_att_mat = soft_max_att_mat / exp_sum.unsqueeze(dim=1).expand_as(soft_max_att_mat)
        topic_vec_mat = (topic_qs_vecs.mul(soft_max_att_mat.unsqueeze(dim=2))).sum(dim=1)
        qa_topic_vecs = torch.zeros(len(set([int(a) for (a, _, _) in topic_info])), topic_qs_vecs.size()[-1]).float()
        if cuda:
            qa_topic_vecs = qa_topic_vecs.cuda()

        q_i = 0
        q_index = topic_info[0][0]
        start = 0
        for t_index, t_info in enumerate(topic_info):
            if q_index != t_info[0]:
                q_index = t_info[0]
                qa_topic_vecs[q_i] = F.normalize(torch.sum(topic_vec_mat[start:t_index], dim=0), p=2, dim=0)
                start = t_index
                q_i += 1
        qa_topic_vecs[q_i] = F.normalize(torch.sum(topic_vec_mat[start:], dim=0), p=2, dim=0)
        return qa_topic_vecs

    def rebuild_user_vec_new(self, cuda, user_qs_vecs, att_vec_list, user_info, qa_vecs):
        if cuda:
            att_list = Variable(torch.FloatTensor(att_vec_list.size()[0], 1).fill_(0).cuda())
        else:
            att_list = Variable(torch.FloatTensor(att_vec_list.size()[0], 1).fill_(0))
        if len(user_info) > 0:
            qa_index = user_info[0][0]
        else:
            return
        qa_i = 0
        att_vec_index = 0
        for u_i, u_info in enumerate(user_info):
            if u_i > 0:
                if u_info[0] != qa_index:
                    qa_index = u_info[0]
                    qa_i += 1
            for u_q_i in range(user_qs_vecs.size()[1]):
                ppp = att_vec_list[att_vec_index].unsqueeze(0).mm(qa_vecs[qa_i].unsqueeze(1)).squeeze(1)
                att_list[att_vec_index] = ppp
                att_vec_index += 1

        att_list_e = torch.exp(att_list).view(user_qs_vecs.size()[0], user_qs_vecs.size()[1], -1)

        soft_max_att_mat = torch.zeros(len(user_info), max([c for (_, _, c) in user_info])).float()
        if cuda:
            soft_max_att_mat = soft_max_att_mat.cuda()
        for i, u_info in enumerate(user_info):
            for j in range(u_info[2]):
                soft_max_att_mat[i][j] = att_list_e[i][j]

        exp_sum = soft_max_att_mat.sum(dim=1)

        soft_max_att_mat = soft_max_att_mat / exp_sum.unsqueeze(dim=1).expand_as(soft_max_att_mat)

        user_vec_mat = (user_qs_vecs.mul(soft_max_att_mat.unsqueeze(dim=2))).sum(dim=1)

        qa_user_vecs = torch.zeros(qa_vecs.size()[0] + len(user_info), user_qs_vecs.size()[-1]).float()
        if cuda:
            qa_user_vecs = qa_user_vecs.cuda()
        sent_index = 1
        now_q_start_index = 0
        for u_i, u_info in enumerate(user_info):
            qa_user_vecs[sent_index] = F.normalize(user_vec_mat[u_i], p=2, dim=0)
            if u_i < len(user_info) - 1:
                if user_info[u_i][0] != user_info[u_i + 1][0]:
                    sent_index += 1
                    qa_user_vecs[now_q_start_index] = F.normalize(qa_user_vecs[now_q_start_index:sent_index].sum(dim=0),
                                                                  p=2, dim=0)
                    now_q_start_index = sent_index
            sent_index += 1
        qa_user_vecs[now_q_start_index] = F.normalize(qa_user_vecs[now_q_start_index:].sum(dim=0), p=2, dim=0)


        return qa_user_vecs

    def rebuild_topic_vec_top_att2(self, cuda, topic_qs_vecs, att_vec_list, topic_info, qa_vecs, hard=1):
        if cuda:
            att_list = Variable(torch.FloatTensor(att_vec_list.size()[0], 1).fill_(0).cuda())
        else:
            att_list = Variable(torch.FloatTensor(att_vec_list.size()[0], 1).fill_(0))
        if len(topic_info) > 0:
            qa_index = topic_info[0][0]
        else:
            return
        qa_i = 0
        att_vec_index = 0
        for t_i, t_info in enumerate(topic_info):
            if t_i > 0:
                if t_info[0] != qa_index:
                    qa_index = t_info[0]
                    qa_i += 1
            for t_q_i in range(topic_qs_vecs.size()[1]):
                ppp = att_vec_list[att_vec_index].unsqueeze(0).mm(qa_vecs[qa_i].unsqueeze(1)).squeeze(1)
                att_list[att_vec_index] = ppp
                att_vec_index += 1
        att_list_e = torch.exp(att_list).view(topic_qs_vecs.size()[0], topic_qs_vecs.size()[1], -1)
        soft_max_att_mat = torch.zeros(len(topic_info), topic_qs_vecs.size()[1]).float()
        if cuda:
            soft_max_att_mat = soft_max_att_mat.cuda()
        for i in range(len(topic_info)):
            if topic_info[i][2] <= hard:
                for j in range(topic_info[i][2]):
                    soft_max_att_mat[i][j] = att_list_e[i][j]
            else:
                max_index = []
                for j in range(topic_info[i][2]):
                    if len(max_index) == 0:
                        max_index.append((j, att_list_e[i][j]))
                    else:
                        shunxu = 0
                        for index, (value_index, value) in enumerate(max_index):
                            if att_list_e[i][j] > value:
                                max_index.insert(shunxu, (j, att_list_e[i][j]))
                                break
                            shunxu += 1
                        if shunxu == len(max_index):
                            max_index.append((j, att_list_e[i][j]))
                        if len(max_index) > hard:
                            max_index.pop()
                for index, (value_index, value) in enumerate(max_index):
                    soft_max_att_mat[i][value_index] = att_list_e[i][value_index]

        exp_sum = soft_max_att_mat.sum(dim=1)
        soft_max_att_mat = soft_max_att_mat / exp_sum.unsqueeze(dim=1).expand_as(soft_max_att_mat)
        topic_vec_mat = (topic_qs_vecs.mul(soft_max_att_mat.unsqueeze(dim=2))).sum(dim=1)
        qa_topic_vecs = torch.zeros(len(set([int(a) for (a, _, _) in topic_info])), topic_qs_vecs.size()[-1]).float()
        if cuda:
            qa_topic_vecs = qa_topic_vecs.cuda()

        q_i = 0
        q_index = topic_info[0][0]
        start = 0
        for t_index, t_info in enumerate(topic_info):
            if q_index != t_info[0]:
                q_index = t_info[0]
                qa_topic_vecs[q_i] = F.normalize(torch.sum(topic_vec_mat[start:t_index], dim=0), p=2, dim=0)
                start = t_index
                q_i += 1
        qa_topic_vecs[q_i] = F.normalize(torch.sum(topic_vec_mat[start:], dim=0), p=2, dim=0)
        return qa_topic_vecs

    def rebuild_user_vec_top_att2(self, cuda, user_qs_vecs, att_vec_list, user_info, qa_vecs, hard=1):
        if cuda:
            att_list = Variable(torch.FloatTensor(att_vec_list.size()[0], 1).fill_(0).cuda())
        else:
            att_list = Variable(torch.FloatTensor(att_vec_list.size()[0], 1).fill_(0))
        if len(user_info) > 0:
            qa_index = user_info[0][0]
        else:
            return
        qa_i = 0
        att_vec_index = 0
        for u_i, u_info in enumerate(user_info):
            if u_i > 0:
                if u_info[0] != qa_index:
                    qa_index = u_info[0]
                    qa_i += 1
            for u_q_i in range(user_qs_vecs.size()[1]):
                ppp = att_vec_list[att_vec_index].unsqueeze(0).mm(qa_vecs[qa_i].unsqueeze(1)).squeeze(1)
                att_list[att_vec_index] = ppp
                att_vec_index += 1

        att_list_e = torch.exp(att_list).view(user_qs_vecs.size()[0], user_qs_vecs.size()[1], -1)

        soft_max_att_mat = torch.zeros(len(user_info), max([c for (_, _, c) in user_info])).float()
        if cuda:
            soft_max_att_mat = soft_max_att_mat.cuda()

        for i, u_info in enumerate(user_info):
            if u_info[2] <= hard:
                for j in range(u_info[2]):
                    soft_max_att_mat[i][j] = att_list_e[i][j]
            else:
                max_index = []
                for j in range(u_info[2]):
                    if len(max_index) == 0:
                        max_index.append((j, att_list_e[i][j]))
                    else:
                        shunxu = 0
                        for index, (value_index, value) in enumerate(max_index):
                            if att_list_e[i][j] > value:
                                max_index.insert(shunxu, (j, att_list_e[i][j]))
                                break
                            shunxu += 1
                        if shunxu == len(max_index):
                            max_index.append((j, att_list_e[i][j]))
                        if len(max_index) > hard:
                            max_index.pop()
                for index, (value_index, value) in enumerate(max_index):
                    soft_max_att_mat[i][value_index] = att_list_e[i][value_index]
        exp_sum = soft_max_att_mat.sum(dim=1)

        soft_max_att_mat = soft_max_att_mat / exp_sum.unsqueeze(dim=1).expand_as(soft_max_att_mat)

        user_vec_mat = (user_qs_vecs.mul(soft_max_att_mat.unsqueeze(dim=2))).sum(dim=1)

        qa_user_vecs = torch.zeros(qa_vecs.size()[0] + len(user_info), user_qs_vecs.size()[-1]).float()
        if cuda:
            qa_user_vecs = qa_user_vecs.cuda()
        sent_index = 1
        now_q_start_index = 0
        for u_i, u_info in enumerate(user_info):
            qa_user_vecs[sent_index] = F.normalize(user_vec_mat[u_i], p=2, dim=0)
            if u_i < len(user_info) - 1:
                if user_info[u_i][0] != user_info[u_i + 1][0]:
                    sent_index += 1
                    qa_user_vecs[now_q_start_index] = F.normalize(qa_user_vecs[now_q_start_index:sent_index].sum(dim=0),
                                                                  p=2, dim=0)
                    now_q_start_index = sent_index
            sent_index += 1
        qa_user_vecs[now_q_start_index] = F.normalize(qa_user_vecs[now_q_start_index:].sum(dim=0), p=2, dim=0)

        return qa_user_vecs

    def rebuild_topic_vec_top_att3(self, cuda, topic_qs_vecs, att_vec_list, topic_info, qa_vecs, hard=4):
        if cuda:
            att_list = Variable(torch.FloatTensor(att_vec_list.size()[0], 1).fill_(0).cuda())
        else:
            att_list = Variable(torch.FloatTensor(att_vec_list.size()[0], 1).fill_(0))
        if len(topic_info) > 0:
            qa_index = topic_info[0][0]
        else:
            return
        qa_i = 0
        att_vec_index = 0
        for t_i, t_info in enumerate(topic_info):
            if t_i > 0:
                if t_info[0] != qa_index:
                    qa_index = t_info[0]
                    qa_i += 1
            for t_q_i in range(topic_qs_vecs.size()[1]):
                ppp = att_vec_list[att_vec_index]
                att_list[att_vec_index] = ppp
                att_vec_index += 1

        att_list_e = torch.exp(att_list).view(topic_qs_vecs.size()[0], topic_qs_vecs.size()[1], -1)
        soft_max_att_mat = torch.zeros(len(topic_info), topic_qs_vecs.size()[1]).float()
        if cuda:
            soft_max_att_mat = soft_max_att_mat.cuda()
        for i in range(len(topic_info)):
            if topic_info[i][2] <= hard:
                for j in range(topic_info[i][2]):
                    soft_max_att_mat[i][j] = att_list_e[i][j]
            else:
                max_index = []
                for j in range(topic_info[i][2]):
                    if len(max_index) == 0:
                        max_index.append((j, att_list_e[i][j]))
                    else:
                        shunxu = 0
                        for index, (value_index, value) in enumerate(max_index):
                            if att_list_e[i][j] > value:
                                max_index.insert(shunxu, (j, att_list_e[i][j]))
                                break
                            shunxu += 1
                        if shunxu == len(max_index):
                            max_index.append((j, att_list_e[i][j]))
                        if len(max_index) > hard:
                            max_index.pop()
                for index, (value_index, value) in enumerate(max_index):
                    soft_max_att_mat[i][value_index] = att_list_e[i][value_index]
        exp_sum = soft_max_att_mat.sum(dim=1)
        soft_max_att_mat = soft_max_att_mat / exp_sum.unsqueeze(dim=1).expand_as(soft_max_att_mat)
        topic_vec_mat = (topic_qs_vecs.mul(soft_max_att_mat.unsqueeze(dim=2))).sum(dim=1)
        qa_topic_vecs = torch.zeros(len(set([int(a) for (a, _, _) in topic_info])), topic_qs_vecs.size()[-1]).float()
        if cuda:
            qa_topic_vecs = qa_topic_vecs.cuda()

        q_i = 0
        q_index = topic_info[0][0]
        start = 0
        for t_index, t_info in enumerate(topic_info):
            if q_index != t_info[0]:
                q_index = t_info[0]
                qa_topic_vecs[q_i] = F.normalize(torch.sum(topic_vec_mat[start:t_index], dim=0), p=2, dim=0)
                start = t_index
                q_i += 1
        qa_topic_vecs[q_i] = F.normalize(torch.sum(topic_vec_mat[start:], dim=0), p=2, dim=0)
        return qa_topic_vecs

    def rebuild_user_vec_top_att3(self, cuda, user_qs_vecs, att_vec_list, user_info, qa_vecs, hard=4):
        if cuda:
            att_list = Variable(torch.FloatTensor(att_vec_list.size()[0], 1).fill_(0).cuda())
        else:
            att_list = Variable(torch.FloatTensor(att_vec_list.size()[0], 1).fill_(0))
        if len(user_info) > 0:
            qa_index = user_info[0][0]
        else:
            return
        qa_i = 0
        att_vec_index = 0
        for u_i, u_info in enumerate(user_info):
            if u_i > 0:
                if u_info[0] != qa_index:
                    qa_index = u_info[0]
                    qa_i += 1
            for u_q_i in range(user_qs_vecs.size()[1]):
                ppp = att_vec_list[att_vec_index]
                att_list[att_vec_index] = ppp
                att_vec_index += 1

        att_list_e = torch.exp(att_list).view(user_qs_vecs.size()[0], user_qs_vecs.size()[1], -1)

        soft_max_att_mat = torch.zeros(len(user_info), max([c for (_, _, c) in user_info])).float()
        if cuda:
            soft_max_att_mat = soft_max_att_mat.cuda()
        for i, u_info in enumerate(user_info):
            if u_info[2] <= hard:
                for j in range(u_info[2]):
                    soft_max_att_mat[i][j] = att_list_e[i][j]
            else:
                max_index = []
                for j in range(u_info[2]):
                    if len(max_index) == 0:
                        max_index.append((j, att_list_e[i][j]))
                    else:
                        shunxu = 0
                        for index, (value_index, value) in enumerate(max_index):
                            if att_list_e[i][j] > value:
                                max_index.insert(shunxu, (j, att_list_e[i][j]))
                                break
                            shunxu += 1
                        if shunxu == len(max_index):
                            max_index.append((j, att_list_e[i][j]))
                        if len(max_index) > hard:
                            max_index.pop()
                for index, (value_index, value) in enumerate(max_index):
                    soft_max_att_mat[i][value_index] = att_list_e[i][value_index]
        exp_sum = soft_max_att_mat.sum(dim=1)

        soft_max_att_mat = soft_max_att_mat / exp_sum.unsqueeze(dim=1).expand_as(soft_max_att_mat)

        user_vec_mat = (user_qs_vecs.mul(soft_max_att_mat.unsqueeze(dim=2))).sum(dim=1)

        qa_user_vecs = torch.zeros(qa_vecs.size()[0] + len(user_info), user_qs_vecs.size()[-1]).float()
        if cuda:
            qa_user_vecs = qa_user_vecs.cuda()
        sent_index = 1
        now_q_start_index = 0
        for u_i, u_info in enumerate(user_info):
            qa_user_vecs[sent_index] = F.normalize(user_vec_mat[u_i], p=2, dim=0)
            if u_i < len(user_info) - 1:
                if user_info[u_i][0] != user_info[u_i + 1][0]:
                    sent_index += 1
                    qa_user_vecs[now_q_start_index] = F.normalize(qa_user_vecs[now_q_start_index:sent_index].sum(dim=0),
                                                                  p=2, dim=0)
                    now_q_start_index = sent_index
            sent_index += 1
        qa_user_vecs[now_q_start_index] = F.normalize(qa_user_vecs[now_q_start_index:].sum(dim=0), p=2, dim=0)

        return qa_user_vecs

    def rebuild_topic_vec_noneatt(self, cuda, topic_qs_vecs, topic_info):
        if cuda:
            att_list = Variable(torch.FloatTensor(topic_qs_vecs.size()[0]*topic_qs_vecs.size()[1], 1).fill_(0).cuda())
        else:
            att_list = Variable(torch.FloatTensor(topic_qs_vecs.size()[0]*topic_qs_vecs.size()[1], 1).fill_(0))
        if len(topic_info) > 0:
            qa_index = topic_info[0][0]
        else:
            return
        qa_i = 0
        att_vec_index = 0
        for t_i, t_info in enumerate(topic_info):
            if t_i > 0:
                if t_info[0] != qa_index:
                    qa_index = t_info[0]
                    qa_i += 1
            for t_q_i in range(topic_qs_vecs.size()[1]):
                ppp = random.random()
                att_list[att_vec_index] = ppp
                att_vec_index += 1
        att_list_e = torch.exp(att_list).view(topic_qs_vecs.size()[0], topic_qs_vecs.size()[1], -1)
        soft_max_att_mat = torch.zeros(len(topic_info), topic_qs_vecs.size()[1]).float()
        if cuda:
            soft_max_att_mat = soft_max_att_mat.cuda()

        for i in range(len(topic_info)):
            for j in range(topic_info[i][2]):
                soft_max_att_mat[i][j] = att_list_e[i][j]
        exp_sum = soft_max_att_mat.sum(dim=1)

        soft_max_att_mat = soft_max_att_mat / exp_sum.unsqueeze(dim=1).expand_as(soft_max_att_mat)
        topic_vec_mat = (topic_qs_vecs.mul(soft_max_att_mat.unsqueeze(dim=2))).sum(dim=1)
        qa_topic_vecs = torch.zeros(len(set([int(a) for (a, _, _) in topic_info])), topic_qs_vecs.size()[-1]).float()
        if cuda:
            qa_topic_vecs = qa_topic_vecs.cuda()

        q_i = 0
        q_index = topic_info[0][0]
        start = 0
        for t_index, t_info in enumerate(topic_info):
            if q_index != t_info[0]:
                q_index = t_info[0]
                qa_topic_vecs[q_i] = F.normalize(torch.sum(topic_vec_mat[start:t_index], dim=0), p=2, dim=0)
                start = t_index
                q_i += 1
        qa_topic_vecs[q_i] = F.normalize(torch.sum(topic_vec_mat[start:], dim=0), p=2, dim=0)
        return qa_topic_vecs

    def rebuild_user_vec_noneatt(self, cuda, user_qs_vecs, user_info):
        if cuda:
            att_list = Variable(torch.FloatTensor(user_qs_vecs.size()[0]*user_qs_vecs.size()[1], 1).fill_(0).cuda())
        else:
            att_list = Variable(torch.FloatTensor(user_qs_vecs.size()[0]*user_qs_vecs.size()[1], 1).fill_(0))
        if len(user_info) > 0:
            qa_index = user_info[0][0]
        else:
            return
        qa_i = 0
        att_vec_index = 0
        for u_i, u_info in enumerate(user_info):
            if u_i > 0:
                if u_info[0] != qa_index:
                    qa_index = u_info[0]
                    qa_i += 1
            for u_q_i in range(user_qs_vecs.size()[1]):
                ppp = random.random()
                att_list[att_vec_index] = ppp
                att_vec_index += 1

        att_list_e = torch.exp(att_list).view(user_qs_vecs.size()[0], user_qs_vecs.size()[1], -1)

        soft_max_att_mat = torch.zeros(len(user_info), max([c for (_, _, c) in user_info])).float()
        if cuda:
            soft_max_att_mat = soft_max_att_mat.cuda()
        for i, u_info in enumerate(user_info):
            for j in range(u_info[2]):
                soft_max_att_mat[i][j] = att_list_e[i][j]

        exp_sum = soft_max_att_mat.sum(dim=1)

        soft_max_att_mat = soft_max_att_mat / exp_sum.unsqueeze(dim=1).expand_as(soft_max_att_mat)

        user_vec_mat = (user_qs_vecs.mul(soft_max_att_mat.unsqueeze(dim=2))).sum(dim=1)

        qa_user_vecs = torch.zeros(user_qs_vecs.size()[0] + len(user_info), user_qs_vecs.size()[-1]).float()
        if cuda:
            qa_user_vecs = qa_user_vecs.cuda()
        sent_index = 1
        now_q_start_index = 0
        for u_i, u_info in enumerate(user_info):
            qa_user_vecs[sent_index] = F.normalize(user_vec_mat[u_i], p=2, dim=0)
            if u_i < len(user_info) - 1:
                if user_info[u_i][0] != user_info[u_i + 1][0]:
                    sent_index += 1
                    qa_user_vecs[now_q_start_index] = F.normalize(qa_user_vecs[now_q_start_index:sent_index].sum(dim=0),
                                                                  p=2, dim=0)
                    now_q_start_index = sent_index
            sent_index += 1
        qa_user_vecs[now_q_start_index] = F.normalize(qa_user_vecs[now_q_start_index:].sum(dim=0), p=2, dim=0)

        return qa_user_vecs

    def cat_topic_vec_att2(self, cuda, QA_vecs, topic_info, topic_q_vecs_mat):
        mat_size = topic_q_vecs_mat.size()
        topic_sent_count = sum([c for (_, _, c) in topic_info])
        if cuda:
            attention_dataset_mat = torch.FloatTensor(mat_size[0], mat_size[1], mat_size[2]).fill_(0).cuda()
        else:
            attention_dataset_mat = torch.FloatTensor(mat_size[0], mat_size[1], mat_size[2]).fill_(0)

        if len(topic_info) > 0:
            qa_index = topic_info[0][0]
        else:
            return attention_dataset_mat
        qa_i = 0

        for t_i, t_info in enumerate(topic_info):
            if t_i > 0:
                if t_info[0] != qa_index:
                    qa_index = t_info[0]
                    qa_i += 1
            for t_q_i in range(t_info[2]):
                attention_dataset_mat[t_i][t_q_i][:mat_size[2]] = topic_q_vecs_mat[t_i][t_q_i]

        return attention_dataset_mat

    def cat_user_vec_att2(self, cuda, QA_vecs, user_info, user_q_vecs):
        user_size = user_q_vecs.size()
        if cuda:
            attention_dataset = torch.FloatTensor(user_size[0], user_size[1], user_size[2]).fill_(0).cuda()
        else:
            attention_dataset = torch.FloatTensor(user_size[0], user_size[1], user_size[2]).fill_(0)
        qa_vec_index = 0
        user_q_index = 0
        for u_index, u_info in enumerate(user_info):
            for k in range(u_info[2]):
                attention_dataset[u_index][k][:user_size[-1]] = user_q_vecs[u_index][k]
                user_q_index += 1
            if u_index < len(user_info) - 1:
                if u_info[0] != user_info[u_index + 1][0]:
                    qa_vec_index += 1
        return attention_dataset

    def cat_topic_vec_att3(self, cuda, QA_vecs, topic_info, topic_q_vecs_mat):
        mat_size = topic_q_vecs_mat.size()
        topic_sent_count = sum([c for (_, _, c) in topic_info])
        if cuda:
            attention_dataset_mat = torch.FloatTensor(mat_size[0], mat_size[1], mat_size[2]*2).fill_(0).cuda()
        else:
            attention_dataset_mat = torch.FloatTensor(mat_size[0], mat_size[1], mat_size[2]*2).fill_(0)

        if len(topic_info) > 0:
            qa_index = topic_info[0][0]
        else:
            return attention_dataset_mat
        qa_i = 0

        for t_i, t_info in enumerate(topic_info):
            if t_i > 0:
                if t_info[0] != qa_index:
                    qa_index = t_info[0]
                    qa_i += 1
            for t_q_i in range(t_info[2]):
                attention_dataset_mat[t_i][t_q_i][:mat_size[2]] = topic_q_vecs_mat[t_i][t_q_i]
                attention_dataset_mat[t_i][t_q_i][mat_size[2]:] = QA_vecs[qa_i]

        return attention_dataset_mat

    def cat_user_vec_att3(self, cuda, QA_vecs, user_info, user_q_vecs):
        user_size = user_q_vecs.size()
        if cuda:
            attention_dataset = torch.FloatTensor(user_size[0], user_size[1], user_size[2]*2).fill_(0).cuda()
        else:
            attention_dataset = torch.FloatTensor(user_size[0], user_size[1], user_size[2]*2).fill_(0)
        qa_vec_index = 0
        user_q_index = 0
        for u_index, u_info in enumerate(user_info):
            for k in range(u_info[2]):
                attention_dataset[u_index][k][:user_size[-1]] = user_q_vecs[u_index][k]
                attention_dataset[u_index][k][user_size[-1]:] = QA_vecs[qa_vec_index]
                user_q_index += 1
            if u_index < len(user_info) - 1:
                if u_info[0] != user_info[u_index + 1][0]:
                    qa_vec_index += 1
        return attention_dataset

    def buildsumvec(self, linking_res, index_list):
        if not linking_res:
            return None
        res = []
        for qa_index in index_list:
            one_qa_res = []
            for sent_index, sent in enumerate(linking_res[qa_index]):
                if len(sent) == 0:
                    one_qa_res.append(torch.zeros(300))
                else:
                    one_qa_res.append(F.normalize(sum(sent), p=2, dim=0))
            res.append(one_qa_res)
        return res
