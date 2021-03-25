# CQAEL
《Entity Linking in Community Question Answering via Leveraging Meta-Data》

#### Environment
Python 3.7

PyTorch 4.0.x

argparse

#### Run our codes
python ./Source/RunModel.py --use_topic 1 --use_user 1
You can also specify the other training parameters, such as epoch (training epoch), lr (learning rate). More details can be found in ./Source/RunModel.py
#### Data Sets
It is worth noting that, for the limitation of the size of files, we only list 2-3 line data as examples in Entity Dictionary Data Set, Entity Embedding Data Set, and Word Information Data Set.
1. CQAEL Data Set
Filename: cqael_dataset.json
The format of training set, testing set, and validation set :
{

    "questions":[
        {
            "question":"",
            "id":"",
            "user_question":""
            "mentions":[
                {
                    "mention":"",
                    "entity":"",
                    "Candidate":"[candidate_name,candidate_id,popularity]",
                    "Gold_index":""
                }
            ]
            "topics":[
                {
                    "topic_name":"",
                    "topic_question":"[question]"
                }
            ]
            "answers":[
                {
                    "answer":"",
                    "user_question":"",
                    "mentions":[
                        {
                            "mention":"",
                            "entity":"",
                            "Candidate":"[candidate_name,candidate_id,popularity]",
                            "Gold_index":""
                        }
                    ]
                }
            ]
        }
    ]
}

2. Entity Dictionary Data Set
Filename: cqael_entityid.txt
The format of entiti dictionary :
{

    entity_name entity_id
}

3. Entity Embedding Data Set
Filename: cqael_ent2vec.txt
The format of entity embedding :
{

    entity_id   vector
}

4. Word information Data Set
Filename: cqael_wordinfo.txt
The format of word information :
{

    word    count   vector
}  
