# coding=utf-8
# 此py用于主函数加载所需要的数据文件
# 所有的loaded函数谢在这里，返回所需数据结构
import numpy as np
import torch
from collections import OrderedDict
import random
import tools
import torch.nn.functional as F


def load_user_question(user_question_file):
    # user 文件格式 q_id-a_id-user_url-questions
    file = open("../Data/" + user_question_file, "r", encoding="utf-8")
    user_question = {}
    for line in file:
        content = line.strip().split("\t")
        if len(content) < 4:
            print("user info error!")
            exit(1)
        q_id = int(content[0])
        a_id = int(content[1])
        user_url = content[2]
        questions_title = content[3].strip().split(",")
        user_question[(q_id, a_id)] = questions_title
    file.close()
    print("File \"user_question\" ({})had Loaded !".format(user_question_file))
    return user_question


def load_topic_question(topic_question_file):
    # topic 文件格式 qid-[t_url-questions]
    file = open("../Data/" + topic_question_file, "r", encoding="utf-8")
    topic_question = {}
    for line in file:
        content = line.strip().split("\t")
        if len(content) < 2:
            print("topic info error!")
            exit(1)
        q_id = int(content[0])
        t_info = {}
        for topic_index in range(len(content))[1::2]:
            topic_url = content[topic_index]
            topic_q_str = content[topic_index + 1].strip().split(",")
            t_info[topic_url] = topic_q_str
        topic_question[q_id] = t_info

    file.close()
    print("File \"topic_question\" ({})had Loaded !".format(topic_question_file))
    return topic_question


def load_entity_vec(cuda,entity_vec_file):
    file = open("../Data/" + entity_vec_file, "r", encoding="utf-8")
    ent_vec = {}
    while True:
        line = file.readline()
        if not line:
            break
        con = line.strip().split("\t")
        if len(con) != 2:
            print("error")
        else:
            e_vec = [float(dim) for dim in con[1].split(",")]
            if len(e_vec) != 300:
                print("length of vector error !")
            else:
                # print(e_vec)
                if cuda:
                    ent_vec[int(con[0])] = F.normalize(torch.FloatTensor(e_vec).cuda(), p=2, dim=0)
                else:
                    ent_vec[int(con[0])] = F.normalize(torch.FloatTensor(e_vec), p=2, dim=0)
    file.close()
    print("File \"entity_vec\" ({})had Loaded !".format(entity_vec_file))
    return ent_vec


def load_candi_dic(candi_dic_file):
    # 加载候选实体文件，{mention:{}}
    file = open("../Data/" + candi_dic_file, "r", encoding="utf-8")
    dic = {}
    for line in file:
        content = line.strip().split("\t")
        if len(content) >= 4:
            one = {}
            for i in range(len(content))[1::3]:
                one[content[i]] = (content[i + 1], float(content[i + 2]))  # entity:(id,pop)
            dic[content[0]] = one
    file.close()
    print("File \"candi_dic\" ({})had Loaded !".format(candi_dic_file))
    return dic


def load_mention_index(mention_info_file):
    # 加载mention索引文件，格式为
    # question_index-answer_index-mention-entity
    file = open("../Data/" + mention_info_file, "r", encoding="utf-8")
    mentions = {}
    for line in file:
        con = line.strip().split("\t")
        if len(con) < 4:
            print("Mention info error ! ")
        if int(con[0]) in mentions:
            if int(con[1]) in mentions[int(con[0])]:
                mentions[int(con[0])][int(con[1])].append((con[2], con[3]))
            else:
                mentions[int(con[0])][int(con[1])] = [(con[2], con[3])]
        else:
            mentions[int(con[0])] = {int(con[1]): [(con[2], con[3])]}
    print("File \"mention_index\" ({})had Loaded !".format(mention_info_file))
    file.close()
    # print(mentions)
    return mentions


def load_ent_id(ent_id_file):
    file = open("../Data/" + ent_id_file, "r", encoding="utf-8")
    ent_id = {}
    for line in file:
        con = line.strip().split("\t")
        ent_id[con[0]] = con[1]
    file.close()
    print("File \"ent_id\" ({})had Loaded !".format(ent_id_file))
    return ent_id


def load_word_counter(word_counter_file):
    file = open("../Data/" + word_counter_file, "r", encoding="utf-8")
    word_counter = OrderedDict()
    for line in file:
        con = line.strip().split("\t")
        word_counter[con[0]] = int(con[1])
    file.close()
    print("File \"word_counter\" ({})had Loaded !".format(word_counter_file))
    return word_counter


def load_word_vector_weight(word_vector_file):
    weight_list = []
    vec_pad = [random.uniform(-1, 1) for i in range(300)]
    vec_unk = [random.uniform(-1, 1) for i in range(300)]
    weight_list.append(vec_pad)
    weight_list.append(vec_unk)
    file = open("../Data/" + word_vector_file, "r", encoding="utf-8")

    for line in file:
        con = line.strip().split("\t")
        w = con[0]
        vec_s = con[1].split(",")
        if len(vec_s) == 300:
            vec = [float(v) for v in vec_s]
        else:
            vec = [random.uniform(-1, 1) for i in range(300)]
            nor = 0.0
            for v in vec:
                nor += v * v
            vec = [v / nor for v in vec]
        weight_list.append(vec)
    file.close()
    weight = torch.FloatTensor(weight_list)
    print("File \"word_vector\" ({})had Loaded !".format(word_vector_file))
    return weight


def load_word_info(word_info_file):
    weight_list = []
    vec_pad = np.random.randn(300)
    vec_unk = np.random.randn(300)

    weight_list.append(vec_pad.tolist())
    weight_list.append(vec_unk.tolist())

    file = open("../Data/" + word_info_file, "r", encoding="utf-8")
    word_counter = []
    for line in file:
        con = line.strip().split("\t")
        w = con[0]
        c = int(con[1])
        word_counter.append((w, c))
        vec_s = con[2].split(",")
        if len(vec_s) == 300:
            vec = [float(v) for v in vec_s]
        weight_list.append(vec)
    file.close()
    weight = torch.FloatTensor(weight_list)
    weight[0] /= torch.norm(weight[0], 2)
    # print(weight[0])
    weight[1] /= torch.norm(weight[1], 2)

    print("File \"word_vector\" ({})had Loaded !".format(word_info_file))
    return weight, word_counter


def get_entity_vec(ent_vec_dic, ent_id):
    if ent_id == None:
        return None
    if ent_id in ent_vec_dic:
        return ent_vec_dic[ent_id]
    else:
        return None


def get_user_question(user_question, question_id, answer_id):
    return user_question[(int(question_id), int(answer_id))]  # 参数是元组，返回值是[quanstion]


def get_topic_question(topic_question, q_id):
    return topic_question[int(q_id)]  # 返回数值是一个字典{topic_url:[question]},应用时遍历字典


def get_candidate(candi_dic, mention):
    if mention in candi_dic:
        return candi_dic[mention]
    else:
        return None


def get_mention_by_index(mention_info, question_index, answer_index):
    if question_index in mention_info:
        if answer_index in mention_info[question_index]:
            return mention_info[question_index][answer_index]
    # print("Get mention error !")
    return None
