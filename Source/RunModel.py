# coding=utf-8
import sys
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from torch import optim
import numpy as np
from random import choice
from collections import OrderedDict, Counter 
from tqdm import tqdm
from torch.autograd import Variable
import itertools
import json
import pickle as pkl
import tools
import datetime
import Model
import LoadFile as lf
from DataProcess import *
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def check_memory(emb_size, max_sents, max_words, b_size, cuda):
    try:
        s_size = (2, b_size, max_sents, max_words, emb_size)
        d_size = (b_size, max_sents, max_words)
        t = torch.rand(*s_size)
        db = torch.rand(*d_size)
        if cuda:
            db = db.cuda()
            t = t.cuda()
        print("-> Quick memory check : OK\n")
    except Exception as e:
        print(e)
        print("Not enough memory to handle current settings {} ".format(s_size))
        print("Try lowering sentence size and length.")
        sys.exit()


def getalldate(datafile, ent_vec_file, popu=False):
    f = open("../Data/"+datafile, "r", encoding="utf-8")
    qa = []
    qa_index = []
    qa_json = json.load(f)
    for q_index, q in enumerate(qa_json["questions"]):
        qa.append(q)
        qa_index.append(q_index)
    if popu:
        all_popures = []
        all_popures_name = []
        for q_index, q in enumerate(qa_json["questions"]):
            one_qa_res = []
            one_qa_res_name = []
            one_q_res = []
            one_q_res_name = []
            for mae in q["mentions"]:
                if mae["Gold_index"] == -1:
                    continue
                candi = mae["Candidates"].strip().split("\t")
                r = int(candi[1])
                one_q_res.append(lf.get_entity_vec(ent_vec_file, r))
                one_q_res_name.append(candi[0].replace(" ", "_"))
            one_qa_res.append(one_q_res)
            one_qa_res_name.append(one_q_res_name)
            answers = q["answers"]
            for a in answers:
                one_a_res = []
                one_a_res_name = []
                for mae in a["mentions"]:
                    if mae["Gold_index"] == -1:
                        continue
                    candi = mae["Candidates"].strip().split("\t")
                    r = int(candi[1])
                    one_a_res.append(lf.get_entity_vec(ent_vec_file, r))
                    one_a_res_name.append(candi[0].replace(" ", "_"))
                one_qa_res.append(one_a_res)
                one_qa_res_name.append(one_a_res_name)
            all_popures.append(one_qa_res)
            all_popures_name.append(one_qa_res_name)
        return list(itertools.zip_longest(qa, qa_index)), [all_popures_name, all_popures]
    return list(itertools.zip_longest(qa, qa_index)), None


def tuple_batch_builder(cuda, topic, user):
    def build_batch(batch_init):
        qa_set, qa_index = zip(*batch_init)
        qa_count = len(qa_set)

        qa_index = torch.LongTensor(qa_index)
        q_len = torch.Tensor([len(q["answers"]) + 1 for q_index, q in enumerate(qa_set)]).long() 

        topic_info = []
        user_info = []

        mention_info = []

        qa_sent_set = []
        topic_sent_set = []
        user_sent_set = []

        for i, qa in enumerate(qa_set):
            qa_sent = []
            qa_sent.append(qa["question_title"])
            qa_topic_count = len(qa["topics"])
            qs_index = 0
            mention = qa["mentions"]
            qa_mention_list = []
            for m in mention:
                if m["Gold_index"] != -1:
                    candi_list = m["Candidates"].strip().split("\t")
                    if cuda:
                        candi_info = \
                            [(
                                candi_list[j], int(candi_list[j + 1]),
                                torch.FloatTensor([float(candi_list[j + 2])]).cuda())
                                for j in range(len(candi_list))[::3]]
                    else:
                        candi_info = [
                            (candi_list[j], int(candi_list[j + 1]), torch.FloatTensor([float(candi_list[j + 2])])) for
                            j in range(len(candi_list))[::3]]
                    qa_mention_list.append((m["mention"], m["entity"], candi_info))
            mention_info.append((qa_index[i], qs_index, qa_mention_list))
            for t in qa["topics"]:
                topic_sent = []
                t_qs = t["topic_question"].strip().split(",")  #
                topic_sent_count = len(t_qs)
                topic_info.append((qa_index[i], qa_topic_count, topic_sent_count))
                for sent in t_qs:
                    topic_sent.append(sent.strip())
                topic_sent_set.append(topic_sent)
            answers = qa["answers"]
            for a in answers:
                qs_index += 1
                qa_sent.append(a["answer_content"])
                mention = a["mentions"]

                qa_mention_list = []
                for m in mention:
                    if m["Gold_index"] != -1:
                        candi_list = m["Candidates"].strip().split("\t")
                        if cuda:
                            candi_info = [(candi_list[j], int(candi_list[j + 1]),
                                           torch.FloatTensor([float(candi_list[j + 2])]).cuda())
                                          for j in range(len(candi_list))[::3]]
                        else:
                            candi_info = [
                                (candi_list[j], int(candi_list[j + 1]), torch.FloatTensor([float(candi_list[j + 2])]))
                                for j in range(len(candi_list))[::3]]
                        qa_mention_list.append((m["mention"], m["entity"], candi_info))
                mention_info.append((qa_index[i], qs_index, qa_mention_list))

                u_qs = a["user_question"].strip().split(",")
                user_sent_count = len(u_qs)
                user_info.append((qa_index[i], len(answers), user_sent_count))
                user_sent = []
                for sent in u_qs:
                    user_sent.append(sent.strip())
                user_sent_set.append(user_sent)
            qa_sent_set.append(qa_sent)

        if user and topic:
            return qa_sent_set, topic_sent_set, user_sent_set, qa_index, q_len, topic_info, user_info, mention_info
        elif topic:
            return qa_sent_set, topic_sent_set, [], qa_index, q_len, topic_info, [], mention_info
        elif user:
            return qa_sent_set, [], user_sent_set, qa_index, q_len, [], user_info, mention_info
        else:
            return qa_sent_set, [], [], qa_index, q_len, [], [], mention_info  # 8个返回值

    return build_batch


def rebuildbatch(vectorizer, qa_index, qa_sent_set, topic_sent_set, user_sent_set, last_res):
    if last_res:
        for q_index, qa_i in enumerate(qa_index):
            for sent_index, last_one_sent_res in enumerate(last_res[0][qa_i]):
                qa_sent_set[q_index][sent_index] = qa_sent_set[q_index][sent_index] + " " + " ".join(last_one_sent_res)
    if topic_sent_set and user_sent_set:
        qa_all_data = vectorizer.vectorizeQA_with_topic_user(qa_sent_set, topic_sent_set, user_sent_set)
    elif topic_sent_set:
        qa_all_data = vectorizer.vectorizeQA_with_topic_user(qa_sent_set, topic_sent_set, [])
    elif user_sent_set:
        qa_all_data = vectorizer.vectorizeQA_with_topic_user(qa_sent_set, [], user_sent_set)
    else:
        qa_all_data = vectorizer.vectorizeQA_with_topic_user(qa_sent_set, [], [])

    info = sorted(
        [(len(sent), len(d), d_index, s_index, sent) for d_index, d in enumerate(qa_all_data) for s_index, sent in
         enumerate(d)], reverse=True)
    batch_data = torch.zeros(len(info), info[0][0]).long()

    for s_index, s in enumerate(info):
        for w_index, w in enumerate(s[-1]):
            batch_data[s_index][w_index] = w
    info = [(ls, lq, qn, sn) for ls, lq, qn, sn, _ in info]

    return batch_data, info


def tuple2var(tensors, data):
    def copy2tensor(t, data):
        t.resize_(data.size()).copy_(data)
        return Variable(t)

    return tuple(map(copy2tensor, tensors, data))


def new_tensors(cuda, types={}):
    ten = []
    for item, t in types.items():
        x = torch.Tensor()
        if t:
            x = x.type(t)
        if cuda:
            x = x.cuda()
        ten.append(x)
    return tuple(ten)


def train(epoch, net, optimizer, dataloader, criterion, cuda, vectorizer, ent_vec, ent_id, res_file, topic, user,
          last_res):
    epoch_loss = 0

    count_all = 0
    right_count_all = 0
    right_per = 0
    right_per_all = 0

    tensor_data = new_tensors(cuda, types={1: torch.LongTensor, 2: torch.LongTensor, 3: torch.LongTensor})

    with tqdm(total=len(dataloader), desc="Evaluating") as pbar:
        for batch_index, (qa_sent_set, topic_sent_set, user_sent_set,
                          qa_index, q_len, topic_info, user_info, mention_info) in enumerate(dataloader):

            batch_data, info = rebuildbatch(vectorizer, qa_index, qa_sent_set, topic_sent_set, user_sent_set, last_res)

            data = tuple2var(tensor_data, (batch_data, qa_index, q_len))
            optimizer.zero_grad()  # 优化器梯度清零
            time1 = datetime.datetime.now()
            out_score, out_info, no_candi_count = net(cuda, data[0], info,
                                                      list(zip(data[1].tolist(), data[2].tolist())), ent_vec, ent_id,
                                                      topic_info, user_info, mention_info, topic, user, None)
            right_count, now_count, loss_dataset = accuracy_with_lossdata(out_score, out_info, ent_vec, last_res)
            print("No candi ：{}. Right :{}. Linked mention :{}.".format(no_candi_count, right_count, now_count))

            right_count_all += right_count
            count_all += (now_count + no_candi_count)
            right_per = right_count / (now_count + no_candi_count)
            right_per_all = right_count_all / count_all

            if loss_dataset!=[] and len(loss_dataset)!=0:
                x1 = torch.cat(([d[0].unsqueeze(dim=0) for d in loss_dataset]), dim=0).float()
                x2 = torch.cat(([d[1].unsqueeze(dim=0) for d in loss_dataset]), dim=0).float()
                y = torch.LongTensor([d[2] for d in loss_dataset]).unsqueeze(dim=0).t().float()
                if cuda:
                    x1 = x1.cuda()
                    x2 = x2.cuda()
                    y = y.cuda()
                loss = criterion(x1, x2, y)
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
            pbar.update(1)
            pbar.set_postfix(
                {"epoch num": epoch, "Batch_Acc": right_per, "All_Acc": right_per_all,
                 "Loss:": epoch_loss / (batch_index + 1)})
        right_per_all = right_count_all / count_all
        print("Finish epoch {} Train. Accuracy is {} !. The value of loss is {:.4f}".format(
            epoch, right_per_all,(epoch_loss / len(dataloader))))
        tools.adddata(res_file + ".txt",
                      "Finish epoch {} Train. Accuracy is {} !. The value of loss is {:.4f}".format(
                          epoch,right_per_all,(epoch_loss / len(dataloader))))

def validation(epoch, net, dataloader,criterion, cuda, vectorizer, ent_vec, ent_id, res_file, topic, user, last_res):
    epoch_loss = 0
    count_all = 0 
    right_count_all = 0
    tensor_data = new_tensors(cuda, types={1: torch.LongTensor, 2: torch.LongTensor, 3: torch.LongTensor})

    with tqdm(total=len(dataloader), desc="Evaluating") as pbar:
        for batch_index, (qa_sent_set, topic_sent_set, user_sent_set,
                          qa_index, q_len, topic_info, user_info, mention_info) in enumerate(dataloader):

            batch_data, info = rebuildbatch(vectorizer, qa_index, qa_sent_set, topic_sent_set, user_sent_set, last_res)

            data = tuple2var(tensor_data, (batch_data, qa_index, q_len))
            out_score, out_info, no_candi_count = net(cuda, data[0], info,
                                                      list(zip(data[1].tolist(), data[2].tolist())), ent_vec, ent_id,
                                                      topic_info, user_info, mention_info, topic, user, None)

            right_count, now_count, loss_dataset = accuracy_with_lossdata(out_score, out_info, ent_vec, last_res)
            print("No candi ：{}. Right :{}. Linked mention :{}.".format(no_candi_count, right_count, now_count))

            right_count_all += right_count
            count_all += (now_count + no_candi_count)
            right_per = right_count / (now_count + no_candi_count)
            right_per_all = right_count_all / count_all

            x1 = torch.cat(([d[0].unsqueeze(dim=0) for d in loss_dataset]), dim=0).float()
            x2 = torch.cat(([d[1].unsqueeze(dim=0) for d in loss_dataset]), dim=0).float()
            y = torch.LongTensor([d[2] for d in loss_dataset]).unsqueeze(dim=0).t().float()
            if cuda:
                x1 = x1.cuda()
                x2 = x2.cuda()
                y = y.cuda()
            loss = criterion(x1, x2, y)

            epoch_loss += loss.item()

            pbar.update(1)
            pbar.set_postfix(
                {"epoch num": epoch, "Batch_Acc": right_per, "All_Acc": right_per_all,
                 "Loss:": epoch_loss / (batch_index + 1)})

        right_per_all = right_count_all / count_all
        print("Finish epoch {} Validation. Accuracy is {} !. The value of loss is {:.4f}".format(
            epoch, right_per_all, (epoch_loss / len(dataloader))))
        tools.adddata(res_file + "vali.txt",
                      "Finish epoch {} Validation. Accuracy is {} !. The value of loss is {:.4f}".format(
                          epoch, right_per_all, (epoch_loss / len(dataloader))))
    return epoch_loss

def test(epoch, net, dataloader, cuda, vectorizer, ent_vec, ent_id, res_file, topic, user, last_res):
    count_all = 0
    right_count_all = 0
    right_per = 0
    right_per_all = 0
    tensor_data = new_tensors(cuda, types={1: torch.LongTensor, 2: torch.LongTensor, 3: torch.LongTensor})

    with tqdm(total=len(dataloader), desc="Evaluating") as pbar:
        for batch_index, (qa_sent_set, topic_sent_set, user_sent_set,
                          qa_index, q_len, topic_info, user_info, mention_info) in enumerate(dataloader):
            batch_data, info = rebuildbatch(vectorizer, qa_index, qa_sent_set, topic_sent_set, user_sent_set, last_res)
            data = tuple2var(tensor_data, (batch_data, qa_index, q_len))
            out_score, out_info, no_candi_count = net(cuda, data[0], info,
                                                      list(zip(data[1].tolist(), data[2].tolist())),
                                                      ent_vec, ent_id, topic_info, user_info, mention_info, topic, user,
                                                      None)
            right_count, now_count, loss_dataset = accuracy_with_lossdata(out_score, out_info, ent_vec, last_res)
            print("No candi ：{}. Right :{}. Linked mention :{}.".format(no_candi_count, right_count, now_count))

            right_count_all += right_count
            count_all += (now_count + no_candi_count)
            right_per = right_count / (now_count + no_candi_count)
            right_per_all = right_count_all / count_all

            pbar.update(1)
            pbar.set_postfix({"epoch num": epoch, "Batch_Acc": right_per, "All_Acc": right_per_all})

        right_per_all = right_count_all / count_all
        print("Finish epoch {} Test. Accuracy is {} ! ".format(epoch, right_per_all))
        tools.adddata(res_file + ".txt", "Finish epoch {} Test. Accuracy is {} ! ".format(epoch, right_per_all))


def save(net, dic, path):
    dict_m = net.state_dict()
    dict_m["word_index"] = dic
    dict_m["reviews"] = torch.Tensor()
    dict_m["topics"] = torch.Tensor()
    dict_m["users"] = torch.Tensor()
    dict_m["word.mask"] = torch.Tensor()
    dict_m["sent.mask"] = torch.Tensor()

    torch.save(dict_m, path)


def accuracy_with_lossdata(scores, info, ent_vec, last_res):
    scores_index = 0

    linking_data_set = []

    mention_index_old = (-1, -1, -1)

    for info_index, (q_index, s_index, m_index, c_id, c, flag) in enumerate(info):
        mention_index_new = (q_index, s_index, m_index)
        if mention_index_new == mention_index_old:
            linking_data_set[-1].append([scores[scores_index], q_index, s_index, m_index, c_id, c, flag])
        else:
            mention_index_old = mention_index_new
            mention_linking_data = [[scores[scores_index], q_index, s_index, m_index, c_id, c, flag]]

            linking_data_set.append(mention_linking_data)
        scores_index += 1
    real_count = len(linking_data_set)
    linking_result_set, right_count = linking(linking_data_set, ent_vec, last_res)

    loss_dataset = build_loss_data(linking_result_set)
    return right_count, real_count, loss_dataset


def linking(linking_data, ent_vec, last_res):
    right_count = 0
    linking_result_set = []
    for m_data in linking_data:
        m_result = list(
            reversed(sorted([[feature, c_id, c, flag] for
                             c_index, (feature, q_index, s_index, m_index, c_id, c, flag) in
                             enumerate(m_data)], reverse=False)))
        if m_result[0][-1] == True:
            right_count += 1
        if last_res:
            linked_ent_vec = lf.get_entity_vec(ent_vec, m_result[0][-3])
            last_res[1][m_data[0][1]][m_data[0][2]][m_data[0][3]] = linked_ent_vec
            last_res[0][m_data[0][1]][m_data[0][2]][m_data[0][3]] = m_result[0][-2].replace(" ", "_")
        linking_result_set.append(m_result)

    return linking_result_set, right_count


def build_loss_data(linked_data_set):
    loss_dataset = []
    for mention_linked in linked_data_set:
        if len(mention_linked) == 1:
            continue
        has_golden_flag = False
        golden_index = -1
        for result_index, candi_info in enumerate(mention_linked):
            if candi_info[-1] == True:
                has_golden_flag = True
                golden_index = result_index
        if has_golden_flag:
            golden_info = mention_linked[golden_index]
            for result_index, candi_info in enumerate(mention_linked):
                if result_index < golden_index:
                    loss_dataset.append([candi_info[0], golden_info[0], -1])
                elif result_index > golden_index:
                    loss_dataset.append([golden_info[0], candi_info[0], 1])
                else:
                    continue
    return loss_dataset


def main(args):
    print("*" * 32 + "Question-Answer EntityLinking By HAN Start！" + "*" * 32)

    print("\nLoading Data:\n" + 25 * "-")

    ent_vec = lf.load_entity_vec(args.cuda, args.ent_vec_file)
    ent_id = lf.load_ent_id(args.ent_id_file)
    word_vec_weight, word_counter = lf.load_word_info(args.word_info_file)
    alldata, popu_res = getalldate(args.data_file, ent_vec,popu=args.global_model)
    print("all of data size if {}".format(len(alldata)))

    topic = args.use_topic
    user = args.use_user

    if not args.vali:
        train_set, test_set = BuildDataSet.build_train_test(alldata, args.split_index)
        print("train_set size is {}".format(len(train_set)))
        print("test_set size is {}".format(len(test_set)))
    else:
        train_set, vali_set, test_set = BuildDataSet.build_train_vali_test(alldata, args.split_index)
        print("train_set size is {}".format(len(train_set)))
        print("validation_set size is {}".format(len(vali_set)))
        print("test_set size is {}".format(len(test_set)))

    vectorizer = Vectorizers(max_word_len=args.max_words, max_sent_len=args.max_sents)

    if args.load:
        print("直接加载训练完成的模型")
        print("model load..!")
        state_dict = torch.load("../Data/snapshot")
        vectorizer.word_index = state_dict["word_index"]
        state_dict.pop("word_index")
        net = Model.HierarchicalQANet(len_word_list=len(vectorizer.word_index), hid_s=150, emb_s=300)
        net.load_state_dict(state_dict)
    else:
        if args.emb:
            print("init model to train..!")
            print("emb load..!")
            print("加载训练好的embedding")
            print(25 * "-" + "\nBuilding word vectors: \n" + "-" * 25)
            vectorizer.wordindex_from_wordcounter(word_counter)
            net = Model.HierarchicalQANet(len_word_list=len(vectorizer.word_index), hid_s=150, emb_s=300)
            net.set_emb_tensor(word_vec_weight)
        else:
            print("init model to train..!")
            print(25 * "-" + "\nBuilding word vectors: \n" + "-" * 25)
            vectorizer.wordindex(train_set.field_iter(0), args.max_feat)
            net = Model.HierarchicalQANet(len_word_list=len(vectorizer.word_index), hid_s=150, emb_s=300)

    if not args.vali:
        batch_builder = tuple_batch_builder(args.cuda, topic, user)
        test_batch_builder = tuple_batch_builder(args.cuda, topic, user)
        dataloader = DataLoader(train_set, batch_size=args.batch_size, collate_fn=batch_builder, shuffle=True,
                                num_workers=0, pin_memory=False)
        test_dataloader = DataLoader(test_set, batch_size=args.batch_size, collate_fn=test_batch_builder, shuffle=True,
                                     num_workers=0, pin_memory=False)
    else:
        batch_builder = tuple_batch_builder(args.cuda, topic, user)
        test_batch_builder = tuple_batch_builder(args.cuda, topic, user)
        vali_builder = tuple_batch_builder(args.cuda, topic, user)

        dataloader = DataLoader(train_set, batch_size=args.batch_size, collate_fn=batch_builder, shuffle=True,
                                num_workers=0, pin_memory=False)
        vali_dataloader = DataLoader(vali_set, batch_size=args.batch_size, collate_fn=vali_builder, shuffle=True,
                                     num_workers=0, pin_memory=False)
        test_dataloader = DataLoader(test_set, batch_size=args.batch_size, collate_fn=test_batch_builder, shuffle=True,
                                     num_workers=0, pin_memory=False)
    criterion = torch.nn.MarginRankingLoss(margin=0.5, size_average=False)
    if args.cuda:
        net.cuda()
    check_memory(args.max_sents, args.max_words, net.emb_size, args.batch_size, args.cuda)
    optimizer = optim.SGD(net.parameters(), lr=1e-5, momentum=0.9)
    torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip_grad)
    print("DataLoader had finished! Start to Training...")
    last_res = popu_res
    vali_loss_record=[]
    for epoch in range(args.epochs):

        train(epoch, net, optimizer, dataloader, criterion, args.cuda, vectorizer, ent_vec, ent_id, args.res_file,
              topic, user, last_res)
        up_count = 0
        net.eval()
        if args.vali and (epoch + 0) % 5 == 0:
            if epoch==0:
                os.makedirs("../Data/"+args.res_file)
            now_vali_loss=validation(epoch, net, vali_dataloader, criterion, args.cuda, vectorizer, ent_vec, ent_id,
                                     args.res_file,topic, user, last_res)  # 验证
            if len(vali_loss_record)==5:
                vali_loss_record.pop(0)
            vali_loss_record.append(now_vali_loss)

            if args.snapshot:
                print("snapshot of model saved as {}".format("snapshot"))
                save(net, vectorizer.word_index, "../Data/"+args.res_file+"/snapshot_"+str(epoch))

            if epoch>=450:
                for val_i in range(len(vali_loss_record))[1:]:
                    if vali_loss_record[val_i]>vali_loss_record[val_i-1]:
                        up_count+=1
        test(epoch, net, test_dataloader, args.cuda, vectorizer, ent_vec, ent_id, args.res_file, topic, user,last_res)
        if up_count >= 3:
            break
        net.train()
    if args.save:
        print("model saved to {}".format(args.save))
        save(net, vectorizer.word_dict, args.save)
    print("Finished！!")


if __name__ == '__main__':
    start = datetime.datetime.now()
    print("start time : {}".format(start))
    for i in range(1):
        parser = argparse.ArgumentParser(description='QA Entity Linking using Hierarchical Attention Networks')
        parser.add_argument("--data_file", type=str, default="cqael_dataset.json")
        parser.add_argument("--ent_vec_file", type=str, default="cqael_ent2vec.txt")
        parser.add_argument("--ent_id_file", type=str, default="cqael_entityid.txt")
        parser.add_argument("--word_info_file", type=str, default="cqael_wordinfo.txt")
        parser.add_argument("--emb-size", type=int, default=300)
        parser.add_argument("--hid-size", type=int, default=150)
        parser.add_argument("--batch-size", type=int, default=16)
        parser.add_argument("--max-feat", type=int, default=20000)
        parser.add_argument("--max-words", type=int, default=140)
        parser.add_argument("--max-sents", type=int, default=12)
        parser.add_argument("--epochs", type=int, default=600)
        parser.add_argument("--use_topic", type=bool, default=True)
        parser.add_argument("--use_user", type=int, default=False)
        parser.add_argument("--cuda", type=bool, default=False)
        parser.add_argument("--load", type=bool, default=False)
        parser.add_argument("--emb", type=bool, default=True)
        parser.add_argument("--vali", type=bool, default=True)
        parser.add_argument("--global_model", type=bool, default=False)
        parser.add_argument("--save", type=bool, default=False)
        parser.add_argument("--clip_grad", type=float, default=1.0)
        parser.add_argument("--snapshot", type=bool, default=True)
        parser.add_argument("--split_index", type=int, default=i + 1)
        parser.add_argument("--res_file", type=str, default="0908_res_att2_topic_hard1_1")
        args = parser.parse_args()
        main(args)
        now = datetime.datetime.now()
        print("finish cross {} train Time : {}".format(i, now))
    end = datetime.datetime.now()
    print("end running time : {}".format(end - start))
