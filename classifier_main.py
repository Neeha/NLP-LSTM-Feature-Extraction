# classifier_main.py

import argparse
import sys
import time
from nerdata import *
from utils import *
from collections import Counter
from optimizers import *
from typing import List
import numpy as np
import string
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import torch
import torch.nn as nn
import re
import itertools
from itertools import *

def _parse_args():
    """
    Command-line arguments to the system. --model switches between the main modes you'll need to use. The other arguments
    are provided for convenience.
    :return: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description='trainer.py')
    parser.add_argument('--model', type=str, default='BAD', help='model to run (BAD, CLASSIFIER)')
    parser.add_argument('--train_path', type=str, default='data/train.txt', help='path to train set (you should not need to modify)')
    parser.add_argument('--dev_path', type=str, default='data/dev.txt', help='path to dev set (you should not need to modify)')
    parser.add_argument('--blind_test_path', type=str, default='data/test_tweets.txt', help='path to dev set (you should not need to modify)')
    parser.add_argument('--test_output_path', type=str, default='eng.testb.out', help='output path for test predictions')
    parser.add_argument('--no_run_on_test', dest='run_on_test', default=True, action='store_false', help='skip printing output on the test set')
    args = parser.parse_args()
    return args

def pad_to_length(np_arr, length):
    """
    Forces np_arr to length by either truncation (if longer) or zero-padding (if shorter)
    :param np_arr:
    :param length: Length to pad to
    :return: a new numpy array with the data from np_arr padded to be of length length. If length is less than the
    length of the base array, truncates instead.
    """
    result = np.zeros((length,9))
    dim = min(length,np_arr.shape[0])
    result[0:dim] = np_arr[0:dim]
    return result

class SentimentLSTM(nn.Module):
    

    def __init__(self, output_size, input_size, hidden_dim, n_layers, drop_prob=0.3):
        """
        Initialize the model by setting up the layers.
        """
        super(SentimentLSTM, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        
        # embedding and LSTM layers
        self.input_size = input_size
        self.ffinit = nn.Linear(input_size,hidden_dim)
        nn.init.xavier_uniform_(self.ffinit.weight)

        self.lstm = nn.LSTM(input_size, hidden_dim, dropout=drop_prob, batch_first=True)#, bidirectional=True)
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
        
        self.dropout = nn.Dropout(0.3)
        self.ffmiddle = nn.Linear(hidden_dim, hidden_dim)
        nn.init.xavier_uniform_(self.ffmiddle.weight)

        # linear and sigmoid layer

        self.fc = nn.Linear(hidden_dim, output_size)
        nn.init.xavier_uniform_(self.fc.weight)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        
        batch_size = x.size(0)
        inputs = torch.ones([x.size(0), x.size(1), self.input_size], dtype=torch.float64)

        out, hidden = self.lstm(inputs.float())
        out = self.ffmiddle(out)
        out = self.fc(out)

        #pdb.set_trace()
        
        # sigmoid function
        sig_out = self.sig(out)
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out.mean(dim=1)
        #sig_out = sig_out[:, -1] # get last batch of labels
        
        return sig_out, hidden

class SentimentExample:
    """
    Wraps a sequence of word indices with a 0-1 label (0 = negative, 1 = positive)
    """
    def __init__(self, indexed_words, label: int):
        self.indexed_words = indexed_words
        self.label = label

    def __repr__(self):
        return repr(self.indexed_words) + "; label=" + repr(self.label)

    def get_indexed_words_reversed(self):
        return [self.indexed_words[len(self.indexed_words) - 1 - i] for i in range(0, len (self.indexed_words))]


def compute_tf_idf(infile: str):
    f = open(infile, encoding='iso8859')
    corpus = []
    for line in f:
        fields = line.split("\t")
        corpus.append(fields[1])

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    return X.toarray()


def read_and_index_sentiment_examples(infile: str, indexer: Indexer, add_to_indexer=True, word_counter=None) -> List[SentimentExample]:
    """
    Reads sentiment examples in the format [0 or 1]<TAB>[raw sentence]; tokenizes and indexes the sentence according
    to the vocabulary in indexer.
    :param infile: file to read
    :param indexer: Indexer containing the vocabulary
    :param add_to_indexer: If add_to_indexer is False, replaces unseen words with UNK, otherwise grows the indexer.
    :param word_counter: optionally keeps a tally of how many times each word is seen (mostly for logging purposes).
    :return: A list of SentimentExample objects read from the file
    """
    # f = open(infile, encoding='utf-8')
    f = open(infile, encoding='iso8859')
    exs = []
    tf_idf_vector = compute_tf_idf(infile)
    index = 0
    for line in f:
        if len(line.strip()) > 0:
            fields = line.split("\t")
            # Slightly more robust to reading bad output than int(fields[0])
            label = 0 if "0" in fields[0] else 1
            # print(fields[1])
            sent = fields[1].lower()
            tokenized_cleaned_sent = list(filter(lambda x: x != '', _clean_str(sent).rstrip().split(" ")))
            pos_tags = nltk.pos_tag(tokenized_cleaned_sent)
            if word_counter is not None:
                for word in tokenized_cleaned_sent:
                    word_counter[word] += 1.0
            # print('Tokenized sent:',tokenized_cleaned_sent)
            indexed_sent = []
            for i,word in enumerate(tokenized_cleaned_sent):
                
                word_3 = word[-3:]
                word_2 = word[-2:]
                postag = pos_tags[i]
                
                prev_word = None
                prev_1_word = None
                if i > 1:
                    prev_1_word = tokenized_cleaned_sent[i-2]
                    prev_word = tokenized_cleaned_sent[i-1]
                elif i > 0:
                    prev_word = tokenized_cleaned_sent[i-1]

                next_word = None
                next_1_word = None
                if i < len(tokenized_cleaned_sent)-2:
                    next_word = tokenized_cleaned_sent[i+1]
                    next_1_word = tokenized_cleaned_sent[i+2] 
                elif i < len(tokenized_cleaned_sent)-1:
                    next_word = tokenized_cleaned_sent[i+1]


                tf_idf = tf_idf_vector[index][i]
                # print('tf:',tf_idf)
                features = [word, word[-3:], word[-2:], pos_tags[i], prev_1_word, prev_word, next_word, next_1_word, tf_idf]
                indexed_word = []
                for i in range(len(features)-1):
                    if indexer.contains(features[i]) or add_to_indexer:
                        feat = indexer.add_and_get_index(features[i])
                    else:
                        feat = indexer.index_of("UNK")
                    indexed_word.append(feat)
                indexed_word.append(tf_idf)
                # print(indexed_word)
                indexed_sent.append(indexed_word)
            index += 1
            exs.append(SentimentExample(indexed_sent, label))
    f.close()
    return exs


def _clean_str(string):
    """
    Tokenizes and cleans a string: contractions are broken off from their base words, punctuation is broken out
    into its own token, junk characters are removed, etc. For this corpus, punctuation is already tokenized, so this
    mainly serves to handle contractions (it's) and break up hyphenated words (crime-land => crime - land)
    :param string: the string to tokenize (one sentence, typicall)
    :return: a string with the same content as the input with whitespace where token boundaries should be, so split()
    will tokenize it.
    """
    # string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"[^A-Za-z0-9(),.!?\'\`\-]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\-", " - ", string)
    # We may have introduced double spaces, so collapse these down
    string = re.sub(r"\s{2,}", " ", string)
    return string

# class PersonExample(object):
#     """
#     Data wrapper for a single sentence for person classification, which consists of many individual tokens to classify.

#     Attributes:
#         tokens: the sentence to classify
#         labels: 0 if non-person name, 1 if person name for each token in the sentence
#     """
#     def __init__(self, tokens: List[str], pos: List[str], labels: List[int]):
#         self.tokens = tokens
#         self.pos = pos
#         self.labels = labels

#     def __len__(self):
#         return len(self.tokens)

# def transform_for_classification(ner_exs: List[LabeledSentence]):
#     """
#     :param ner_exs: List of chunk-style NER examples
#     :return: A list of PersonExamples extracted from the NER data
#     # """
#     # for labeled_sent in ner_exs:
#     #     tags = bio_tags_from_chunks(labeled_sent.chunks, len(labeled_sent))
#     #     labels = [1 if tag.endswith("PER") else 0 for tag in tags]
#     #     words = [tok.word for tok in labeled_sent.tokens]
#     #     pos = [tok.pos for tok in labeled_sent.tokens]
#     #     yield PersonExample(words, pos, labels)

#     for labeled_sent in ner_exs:
#         # print(labeled_sent)
#         labels = [labels for labels in labeled_sent.labels]
#         words = [tok.word for tok in labeled_sent.tokens]
#         pos = [tok.pos for tok in labeled_sent.tokens]
#         yield PersonExample(words, pos, labels)
#     print('Transformation done')


# def word2features(sent, i):
#     word = sent.tokens[i].word
#     postag = sent.tokens[i].pos
#     postag = postag[1]
    
#     # features = []
#     features = [word, word.lower(), word[-3:],word[-2:],postag]
     
#     return features

# def get_features(ner_exs: List[LabeledSentence]):

#     #get input features matrix
#     input_features = []
#     pos_list = []
#     for sent in ner_exs:
#         input_feature = []
#         for i in range(len(sent)):
#             if sent.tokens[i].pos not in pos_list:
#                 pos_list.append(sent.tokens[i].pos)
#             input_feature.append(word2features(sent, i))
#         input_features.append(input_feature)
#             # input_features.append(word2features(sent, i))
#     return input_features

# def sent2labels(sent):
#     tags = bio_tags_from_chunks(sent.chunks, len(sent))
#     labels = np.asarray([1 if tag.endswith("PER") else 0 for tag in tags])
#     return labels
    # return [label for token, postag, label in sent]

def get_labels(ner_exs: List[LabeledSentence]):
    labels = []
    for sent in ner_exs:
        labels.append(int(sent.labels[0]))
    return labels

def transform_input(ner_exs: List[LabeledSentence], data, vec):
    X = get_features(ner_exs)
    Y = get_labels(ner_exs)
    if data=='test' or data=='dev':
        X = np.vectorize(X)
    else:
        X = np.vectorize(X)
    return X, Y

# def print_info(ner_exs: List[PersonExample]):
#     for ex in ner_exs:
#         for idx in range(0, len(ex)):
#             if ex.labels[idx] == 1:
#                 print(ex.tokens[idx],'--',ex.pos[idx])

# class CountBasedPersonClassifier(object):
#     """
#     Person classifier that takes counts of how often a word was observed to be the positive and negative class
#     in training, and classifies as positive any tokens which are observed to be positive more than negative.
#     Unknown tokens or ties default to negative.
#     Attributes:
#         pos_counts: how often each token occurred with the label 1 in training
#         neg_counts: how often each token occurred with the label 0 in training
#     """
#     def __init__(self, pos_counts: Counter, neg_counts: Counter):
#         self.pos_counts = pos_counts
#         self.neg_counts = neg_counts

#     def predict(self, tokens: List[str], idx: int):
#         if self.pos_counts[tokens[idx]] > self.neg_counts[tokens[idx]]:
#             return 1
#         else:
#             return 0


# def train_count_based_binary_classifier(ner_exs: List[PersonExample]):
#     """
#     :param ner_exs: training examples to build the count-based classifier from
#     :return: A CountBasedPersonClassifier using counts collected from the given examples
#     """
#     pos_counts = Counter()
#     neg_counts = Counter()
#     for ex in ner_exs:
#         for idx in range(0, len(ex)):
#             if ex.labels[idx] == 1:
#                 pos_counts[ex.tokens[idx]] += 1.0
#             else:
#                 neg_counts[ex.tokens[idx]] += 1.0
#     print(repr(pos_counts))
#     print(repr(pos_counts["Peter"]))
#     print(repr(neg_counts["Peter"]))
#     print(repr(pos_counts["aslkdjtalk;sdjtakl"]))
#     return CountBasedPersonClassifier(pos_counts, neg_counts)


# class PersonClassifier(object):
#     """
#     Classifier to classify a token in a sentence as a PERSON token or not.
#     Constructor arguments are merely suggestions; you're free to change these.
#     """

#     def __init__(self, weights: np.ndarray, indexer: Indexer):
#         self.weights = weights
#         self.indexer = indexer

#     def predict(self, tokens: List[str], idx: int):
#         """
#         Makes a prediction for token at position idx in the given PersonExample
#         :param tokens:
#         :param idx:
#         :return: 0 if not a person token, 1 if a person token
#         """
#     def predict(self, tokens, idx):
#         raise Exception("Implement me!")


    # def orthographic_structure(indexer:Indexer):



def train_classifier(train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> List[SentimentExample]:
# def train_classifier(X, Y_train):
    # clf = LogisticRegression(penalty='l2', dual=False, tol=0.001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, max_iter=100)
    # model = clf.fit(X, Y_train)
    # return model

    seq_max_len = 60
    # print(train_exs[0])
    # To get you started off, we'll pad the training input to 60 words to make it a square matrix.
    train_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in train_exs])
    # Also store the sequence lengths -- this could be useful for training LSTMs
    train_seq_lens = np.array([len(ex.indexed_words) for ex in train_exs])
    # Labels
    train_labels_arr = np.array([ex.label for ex in train_exs])

    # train_mat = np.asarray([np.array(X)])
    # train_labels_arr = np.array(Y_train)

    # get numpy array with sequence length for dev and test
    dev_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in dev_exs])
    # Also store the sequence lengths -- this could be useful for training LSTMs
    dev_seq_lens = np.array([len(ex.indexed_words) for ex in dev_exs])
    # Labels
    dev_labels_arr = np.array([ex.label for ex in dev_exs])

    # test_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in test_exs])
    # # Also store the sequence lengths -- this could be useful for training LSTMs
    # test_seq_lens = np.array([len(ex.indexed_words) for ex in test_exs])
    # # Labels
    # test_labels_arr = np.array([ex.label for ex in test_exs])
    # #print(test_labels_arr)

    from torch.utils.data import TensorDataset, DataLoader
    # create Tensor datasets
    train_data = TensorDataset(torch.from_numpy(train_mat), torch.from_numpy(train_labels_arr))
    dev_data = TensorDataset(torch.from_numpy(dev_mat), torch.from_numpy(dev_labels_arr))
    # test_data = TensorDataset(torch.from_numpy(test_mat), torch.from_numpy(test_labels_arr))
    # dataloaders
    batch_size = 100 #10
    # make sure to SHUFFLE your data
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    dev_loader = DataLoader(dev_data, shuffle=True, batch_size=batch_size)

    output_size = 1
    input_size = 20
    hidden_dim = 512
    n_layers = 2
    net = SentimentLSTM(output_size, input_size, hidden_dim, n_layers)
    print(net)

    # loss and optimization functions
    lr=0.001 #0.001
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # training params

    epochs = 5 #20 

    counter = 0
    print_every = 1
    
    num_correct = 0


    net.train()
    for epoch_iter in range(epochs):
        
        for inputs, labels in train_loader:
            counter += 1


            net.zero_grad()
            output, h = net.forward(inputs)

            # calculate the loss and perform backprop
            loss = criterion(output.squeeze(), labels.float())
            loss.backward()
            
            optimizer.step()

            # loss stats
            if counter % print_every == 0:
                val_losses = []
                net.eval()
                for inputs, labels in train_loader:
                    output, val_h = net(inputs)
                    val_loss = criterion(output.squeeze(), labels.float())

                    val_losses.append(val_loss.item())
                    
                # net.train()
                print("Epoch: {}/{}...".format(epoch_iter+1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.6f}...".format(loss.item()),
                      "Train Loss: {:.6f}".format(np.mean(val_losses)))
                pred = torch.round(output.squeeze())  
                       
                # compare predictions to true label
                correct_tensor = pred.eq(labels.float().view_as(pred))
                correct = np.squeeze(correct_tensor.numpy())
                num_correct += np.sum(correct)
                train_acc = num_correct/batch_size
                print("train accuracy:",train_acc)
                num_correct = 0

    #save model
    torch.save(net.state_dict(),'/home/neeha/UT/Sem3/NLP/FP/Mini 1/mini1-distrib/saved_model_features_rotten.pth')
    dev_losses = []
    num_correct = 0

    
    net.eval()
    for inputs, labels in dev_loader:
        output, val_h = net(inputs)        
        dev_loss = criterion(output.squeeze(), labels.float())
        dev_losses.append(dev_loss.item())       
        pred = torch.round(output.squeeze())         
        correct_tensor = pred.eq(labels.float().view_as(pred))
        correct = np.squeeze(correct_tensor.numpy())
        num_correct += np.sum(correct)

    print("dev final loss: {:.3f}".format(np.mean(dev_losses)))
    print("num_correct", num_correct)
    print("len(dev_loader.dataset):",len(dev_loader.dataset))

    dev_acc = num_correct/len(dev_loader.dataset)
    print("dev final accuracy: {:.3f}".format(dev_acc))

    

def evaluate_classifier(dev_exs: List[SentimentExample], test_exs: List[SentimentExample], model_path) -> List[SentimentExample]:
    # 59 is the max sentence length in the corpus, so let's set this to 60
    seq_max_len = 60
    test_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in test_exs])
    # Also store the sequence lengths -- this could be useful for training LSTMs
    test_seq_lens = np.array([len(ex.indexed_words) for ex in test_exs])
    # Labels
    test_labels_arr = np.array([ex.label for ex in test_exs])
    #/print(test_labels_arr)



    from torch.utils.data import TensorDataset, DataLoader
    test_data = TensorDataset(torch.from_numpy(test_mat), torch.from_numpy(test_labels_arr))
    # dataloaders
    batch_size = 1#100 #10
    # Instantiate the model w/ hyperparams
    output_size = 1
    input_size = 20
    hidden_dim = 512
    n_layers = 2
    net = SentimentLSTM(output_size, input_size, hidden_dim, n_layers)
    print(net)

    net.load_state_dict(torch.load(model_path))


    num_correct = 0


    net.eval()
    # get numpy array with sequence length for dev and test
    dev_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in dev_exs])
    # Also store the sequence lengths -- this could be useful for training LSTMs
    dev_seq_lens = np.array([len(ex.indexed_words) for ex in dev_exs])
    # Labels
    dev_labels_arr = np.array([ex.label for ex in dev_exs])
    dev_data = TensorDataset(torch.from_numpy(dev_mat), torch.from_numpy(dev_labels_arr))
    dev_loader = DataLoader(dev_data, shuffle=True, batch_size=100)


    for inputs, labels in dev_loader:
        output, val_h = net(inputs)
        pred = torch.round(output.squeeze())
        correct_tensor = pred.eq(labels.float().view_as(pred))
        correct = np.squeeze(correct_tensor.numpy())
        num_correct += np.sum(correct)

    print("num_correct", num_correct)
    print("len(dev_loader.dataset):",len(dev_loader.dataset))

    dev_acc = num_correct/len(dev_loader.dataset)
    print("dev final accuracy: {:.3f}".format(dev_acc))



    test_loader = DataLoader(test_data, shuffle=False, batch_size=1)
    pred_test = []
    i = 0
    for inputs, lables in test_loader:
        output, h = net(inputs)
        pred = torch.round(output.squeeze())
        inputs = inputs.numpy()
        #print(inputs.tolist())
        inputs = list(itertools.chain(*inputs))
        pred_test.append(SentimentExample(inputs[:test_seq_lens[i]], int(pred.detach().numpy())))
        i +=1
    return pred_test

def print_evaluation(golds: List[int], predictions: List[int]):
    """
    Prints statistics about accuracy, precision, recall, and F1
    :param golds: list of {0, 1}-valued ground-truth labels for each token in the test set
    :param predictions: list of {0, 1}-valued predictions for each token
    :return:
    """
    num_correct = 0
    num_pos_correct = 0
    num_pred = 0
    num_gold = 0
    num_total = 0
    if len(golds) != len(predictions):
        raise Exception("Mismatched gold/pred lengths: %i / %i" % (len(golds), len(predictions)))
    for idx in range(0, len(golds[0])):
        gold = golds[0][idx]
        prediction = predictions[0][idx]
        if prediction == gold:
            num_correct += 1
        if prediction == 1:
            num_pred += 1
        if gold == 1:
            num_gold += 1
        if prediction == 1 and gold == 1:
            num_pos_correct += 1
        num_total += 1
    print("Accuracy: %i / %i = %f" % (num_correct, num_total, float(num_correct) / num_total))
    prec = float(num_pos_correct) / num_pred if num_pred > 0 else 0.0
    rec = float(num_pos_correct) / num_gold if num_gold > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec > 0 and rec > 0 else 0.0
    print("Precision: %i / %i = %f" % (num_pos_correct, num_pred, prec))
    print("Recall: %i / %i = %f" % (num_pos_correct, num_gold, rec))
    print("F1: %f" % f1)

# def predict_write_output_to_file_bad(exs: List[PersonExample], classifier: PersonClassifier, outfile: str):
#     """
#     Runs prediction on exs and writes the outputs to outfile, one token per line
#     :param exs:
#     :param classifier:
#     :param outfile:
#     :return:
#     """
#     f = open(outfile, 'w')
#     for ex in exs:
#         for idx in range(0, len(ex)):
#             prediction = classifier.predict(ex.tokens, idx)
#             f.write(ex.tokens[idx] + " " + repr(int(prediction)) + "\n")
#         f.write("\n")
#     f.close()

# def predict_write_output_to_file(exs: List[LabeledSentence], X_test, classifier: LogisticRegression, outfile: str):
#     """
#     Runs prediction on exs and writes the outputs to outfile, one token per line
#     :param exs:
#     :param classifier:
#     :param outfile:
#     :return:
#     """
#     f = open(outfile, 'w')
#     predictions = classifier.predict(X_test)
#     j = 0
#     for ex in exs:
#         for idx in range(0, len(ex)):
#             f.write(ex.tokens[idx] + " " + repr(int(predictions[j])) + "\n")
#             j += 1
#         f.write("\n")
#     f.close()

if __name__ == '__main__':
    start_time = time.time()
    args = _parse_args()
    print(args)
    # Load the training and test data
    indexer = Indexer()
    vec = DictVectorizer()
    train_exs = read_and_index_sentiment_examples(args.train_path, indexer)
    dev_exs = read_and_index_sentiment_examples(args.dev_path, indexer)
    test_exs = read_and_index_sentiment_examples(args.blind_test_path, indexer)

    print("Data reading and training took %f seconds" % (time.time() - start_time))
    model_path = '/home/neeha/UT/Sem3/NLP/FP/Mini 1/mini1-distrib/saved_model_features_rotten.pth'
    # Train the model
    if args.model == "BAD":
        classifier = train_count_based_binary_classifier(train_class_exs)
        print("===Train accuracy===")
        evaluate_classifier(train_class_exs, classifier)
        print("===Dev accuracy===")
        evaluate_classifier(dev_class_exs, classifier)

    else:
        classifier = train_classifier(train_exs, dev_exs)
        print("===Train accuracy===")
        print("===Dev accuracy===")
        evaluate_classifier(dev_exs, dev_exs, model_path)
    
    # if args.run_on_test:
    #     print("Running on test")
    #     test_exs = list(transform_for_classification(read_data(args.blind_test_path)))
    #     X_test, Y_test = transform_input(read_data(args.blind_test_path),'test',vec)
    #     predict_write_output_to_file(test_exs, X_test, classifier, args.test_output_path)
    #     # predict_write_output_to_file_bad(test_exs, classifier, args.test_output_path)
    #     print("Wrote predictions on %i labeled sentences to %s" % (len(test_exs), args.test_output_path))



