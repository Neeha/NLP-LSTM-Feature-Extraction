Running the project
python3 classifier_main.py --model CLASSIFIER

Feature length  train : (204567, 136547)
Feature length  dev : (51578, 136547)
===Train accuracy===
Accuracy: 204293 / 204567 = 0.998661
Precision: 10918 / 10982 = 0.994172
Recall: 10918 / 11128 = 0.981129
F1: 0.987607
===Dev accuracy===
Accuracy: 51018 / 51578 = 0.989143
Precision: 2759 / 2929 = 0.941960
Recall: 2759 / 3149 = 0.876151
F1: 0.907864
Running on test
Feature length  test : (46666, 136547)
Wrote predictions on 3684 labeled sentences to eng.testb.out