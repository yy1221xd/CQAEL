from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict, Counter
from tqdm import tqdm
import torch
import tools
class BuildDataSet(Dataset):
    def __init__(self, datalist):
        super(BuildDataSet, self).__init__()
        self.datalist = datalist

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, item):
        return self.datalist[item]

    def field_iter(self, field):
        def get_data():
            for i in range(len(self)):
                yield self[i][field]

        return get_data

    @staticmethod
    def build_train_test(datalist,split_index):
        train = []
        test = []
        for i in range(len(datalist)):
            if (i + split_index) % 5 == 0:
                test.append(datalist[i])
            else:
                train.append(datalist[i])
        return BuildDataSet(train), BuildDataSet(test)

    @staticmethod
    def build_train_vali_test(datalist,split_index):
        
        train = []
        vali=[]
        test = []

        for i in range(len(datalist)):  # 暂定
            if (i + split_index) % 5 == 0:
                test.append(datalist[i])
            elif (i + split_index+1 ) % 5==0:
                vali.append(datalist[i])
            else:
                train.append(datalist[i])
        return BuildDataSet(train), BuildDataSet(vali), BuildDataSet(test)


class Vectorizers():
    def __init__(self, word_index=None, max_sent_len=12, max_word_len=140):
        self.word_index = word_index
        self.max_sent_len = max_sent_len
        self.max_word_len = max_word_len

    def getwordindex(self, data, max_words):
        word_counter = Counter(
            w.lower() for qa in tqdm(data(), desc="Statistics frequence for each word:") for sen in qa.split("\t") for w
            in sen.split(" "))
        word_index = {w: i for i, (w, c) in tqdm(enumerate(word_counter.items(), start=2), desc="Build word index:")}
        word_index["_padding_"] = 0
        word_index["_unk_word_"] = 1
        print("Word index has finished! Including {} words".format(len(word_index)))
        return word_index

    def wordindex(self, data, max_words):
        self.word_index = self.getwordindex(data, max_words)


    def getwordindex_form_wordcounter(self, word_counter):
        word_index = {w: i for i, (w, c) in enumerate(word_counter, start=2)}
        word_index["_padding_"] = 0
        word_index["_unk_word_"] = 1
        print("Word index has finished! Including {} words".format(len(word_index)))
        return word_index

    def wordindex_from_wordcounter(self, word_counter):
        self.word_index = self.getwordindex_form_wordcounter(word_counter)

    def vectorizeQA_with_topic_user(self, QA,topic_Qs,user_Qs):
        Qa_vec=self.vectorizeQs(QA)
        Topic_vec=self.vectorizeQs(topic_Qs)
        User_vec=self.vectorizeQs(user_Qs)
        Qa_vec.extend(Topic_vec)
        Qa_vec.extend(User_vec)
        return Qa_vec

    def vectorizeQA(self, QA): 
        if self.word_index is None:
            print("word index has not loaded! ")
            raise Exception
        qa_all = []
        for q in QA:
            qa = []
            for i, sent in enumerate(q.split("\t")):
                s = []
                for j, w_ in enumerate(sent.split(" ")):
                    w=w_
                    if w in self.word_index:
                        s.append(self.word_index[w])
                    else:
                        s.append(self.word_index["_unk_word_"])
                if len(s) >= 1:
                    qa.append(torch.LongTensor(s))
            if len(qa) == 0:
                qa = torch.LongTensor(self.word_index["_unk_word_"])
            qa_all.append(qa)
        return qa_all

    def vectorizeQs(self, Qs):
        if self.word_index is None:
            print("word index has not loaded! ")
            raise Exception
        qs_all = []
        for q in Qs:
            qs = []
            for i, sent in enumerate(q):
                s = []
                for j, w_ in enumerate(sent.split(" ")):
                    w = w_.lower()
                    if w in self.word_index:
                        s.append(self.word_index[w])
                    else:
                        s.append(self.word_index["_unk_word_"])
                if len(s) >= 1:
                    qs.append(torch.LongTensor(s))
            if len(qs) == 0:
                qs = torch.LongTensor(self.word_index["_unk_word_"])
            qs_all.append(qs)

        return qs_all