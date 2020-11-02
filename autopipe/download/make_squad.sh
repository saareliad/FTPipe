#! /bin/bash
if [ ! -d squad1 ] ; then 
mkdir squad1
cd squad1 || exit 1
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
wget https://github.com/allenai/bi-att-flow/blob/master/squad/evaluate-v1.1.py
cd ..

fi

if [ ! -d squad2 ] ; then 
mkdir squad2
cd squad2 || exit 1
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json
curl https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/ > evaluate-v2.0.py
cd ..
fi
