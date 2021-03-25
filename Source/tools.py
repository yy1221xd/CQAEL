#coding=utf-8
#本py用于一些构建数据文本的函数存储
#也包括一些常用函数及常用代码段
def buildmentionfile(questions,mentionfile):
    #用于使用json构建mention信息，mention信息文本格式为:
    # question_index-answer_index-mention-entity
    res=""
    out_file=open("../Data/"+mentionfile,"w",encoding="utf-8")
    question_count=0
    for q in questions:
        answer_count = 0
        mentions = q["mentions"]
        for mae in mentions:
            e = mae["entity"]
            m=mae["mention"]
            res=res+str(question_count)+"\t"+str(answer_count)+"\t"+m+"\t"+e+"\n"
        answers = q["answers"]
        for a in answers:
            answer_count+=1
            mentions = a["mentions"]
            for mae in mentions:
                e = mae["entity"]
                m = mae["mention"]
                res = res + str(question_count) + "\t" + str(answer_count) + "\t" + m + "\t" + e + "\n"
        question_count+=1
    out_file.write(res)
    out_file.close()

def TraversalQuestions(questions):

    for q in questions:
        answers=q["answers"]
        mentions=q["mentions"]
        for mae in mentions:
            m=mae["mention"]
            e=mae["entity"]
        for a in answers:
            mentions=a["mentions"]
            for mae in mentions:
                m = mae["mention"]
                e = mae["entity"]

def Traversalfile(filename):
    file=open("../Data/"+filename,"r",encoding="utf-8")
    for line in file:
        con=line.strip().split("\t")
    file.close()
    print("File \"filename\" ({})had Loaded !".format(filename))

def printveclist(title,generator):
    print("title : {}".format(title))
    for i,line in enumerate(generator):
        print("{} : {}".format(i,line))
def print10line(filename):
    file = open("../Data/" + filename, "r", encoding="utf-8")
    line_count=0
    while True:
        line =file.readline()
        if not line:
            break
        line_count+=1
        if line_count>=10:
            break
        print(line)
    file.close()

def writedata(generator,filename=None):
    if filename is None:
        print("Need Filename !")
        exit(1)
    file = open("../Data/" + filename+".txt", "a+", encoding="utf-8")
    res=""
    line_count=0
    for line in generator:
        res=res+str(line)+"\n"
        line_count+=1
        if line_count%500==0:
            file.write(res)
            res=""
    file.write(res)
    file.close()
    print("File \"{}\" had writed !".format(filename))

def adddata(filename, content):
    file = open("../Data/" + filename, "a+", encoding="utf-8")
    file.write(content+"\n")
    file.close()

def traversaljson(jsonfile):
    questions = jsonfile["questions"]
    question_index = 0
    for q in questions:
        topics = q["topics"]
        mentions = q["mentions"]
        for mae in mentions:
            m = mae["mention"]
            e = mae["entity"]
        answer_index = 0
        answers = q["answers"]
        for a in answers:
            mentions = a["mentions"]
            for mae in mentions:
                m = mae["mention"]
                e = mae["entity"]
            answer_index += 1
        question_index += 1