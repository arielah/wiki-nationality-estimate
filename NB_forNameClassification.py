#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import numpy as np
from sklearn.metrics import accuracy_score
import random
random.seed(111)
from sklearn.model_selection import train_test_split


# # Download CSV and filter / parse

# In[2]:


#!wget https://github.com/arielah/CIS519_FinalProj/raw/master/annotated_names.tsv
#looks like file is changing. work on local file for now


# In[3]:



# # Functions used for NB algorithms

# In[6]:
def parse_filter_csv(csv, fn_idx = 0, ln_idx = 2, label_idx = 3, 
                     uniqID_idx = 4, delim = ',', max_name_len = 100,
                    ngram=2, split_data = False, frac_train = 0.6,
                    frac_val = 0.2, frac_test = 0.2,num_cols=5):
    
    '''
    INPUT: CSV or TSV with firstName, lastName, and label
    in different columns and 
    
    RETURNS: ordered lists of names, n-grams, and country labels.
    
        If split_data == True:
            returns one of each of the above for train, test, and validation
            with indicated splits
    
    Depending on the file, tell this function where to find:
        fn_idx: column index of first name
        ln_idx: column index of last name
        labe_idx: column index of country label 
        delmim: what to split lines by to get columns
    
    Filters:
        lines with characters from bad_chars (see below)
        names longer than 100 characters
        names with label "Other"
        names with firstName and/or lastName with less than 2 characters
        names that are exact duplicates within a country label 
    
    NOTE: Name lists returned are ordered by accessing dictionary
    keys corresponding to countries. You probably want to shuffle these
    before using them for testing, training, etc. 
    
    TO-DO: keep track of uniqID indexes when/if that is added
    '''
    random.seed(111)
    count_dic = {} 
    bad_chars = ['!', '"', '%', '&', '(', ')', '*', '/',':', ';', 
                 '=', '?', '@',']', '_', '`','{', '|', '}', '~'] 
    for x in range(0,10):
        bad_chars.append(str(x))
    all_lines = 0
    final_lines = 0
    line_lens = []
    name_dic = {}
    bad_char_lines = 0
    all_letters = set([])
    fd = open(csv,'r')
    #print('Header:\n%s'%fd.readline())
    for line in fd:
        bad_char = False
        all_lines += 1
        l = line.strip().split(delim)
        fn = l[fn_idx].strip()
        ln = l[ln_idx].strip()
        country = l[label_idx].strip()
        if country == 'Other':
            continue
        if len(fn) < 2:
            continue
        if len(ln) < 2:
            continue
        name_len = len(fn) + len(ln)
        #check names for characters
        full_name = fn + ' ' + ln
        for char in full_name:
            if char in bad_chars:
                #print('##%s##'%char, line)
                bad_char = True
                bad_char_lines += 1
            else:
                all_letters.add(char)
        if bad_char == True:
            continue
        if name_len > 100: 
            continue
        final_lines += 1
        try:
            count_dic[country] += 1
            name_dic[country].add(full_name)
        except:
            count_dic[country] = 1
            name_dic[country] = set([full_name])
    #print('All lines processed:%s\nLines meeting criteria:%s'%(all_lines, final_lines))
    #print('total bad char lines: %s'%bad_char_lines)
    print('Here are for each country the # of valid names and # uniq valid names:')
    name_list = []
    label_list = []
    doc_list = []
    for c in count_dic.keys():
        print('%s\t%s\t%s'%(c, count_dic[c], len(name_dic[c])))
        for full_name in name_dic[c]:
            nChars = [full_name[i:i+ngram] for i in range(len(full_name)-(ngram-1))]
            doc_list.append(nChars)
            name_list.append(full_name)
            label_list.append(c)
    fd.close()
    print('FINAL NAME COUNT: %s'%len(name_list))
    if split_data == True:
        n_test = int(len(name_list) * frac_test)
        n_val = int(len(name_list) * frac_val)
        #first split test off
        doc_train, doc_test, y_train, y_test, name_train, name_test = train_test_split(doc_list, label_list, name_list, stratify=label_list, test_size=n_test, random_state=1)
        #next split gives 20% to validation
        doc_train, doc_val, y_train, y_val, name_train, name_val = train_test_split(doc_train, y_train, name_train, stratify=y_train, test_size=n_val, random_state=1)
        print('split to train,val, test data with these lengths:')
        print('train: %s, validation: %s, test: %s'%(len(doc_train), len(doc_val), len(doc_test)))
        return doc_train, doc_test, doc_val, y_train, y_test, y_val, name_train, name_test, name_val
    else:
        return name_list, doc_list, label_list



def get_vocabulary(D):
    """
    Given a list of documents, where each document is represented as
    a list of tokens, return the resulting vocabulary. The vocabulary
    should be a set of tokens which appear more than once in the entire
    document collection plus the "<unk>" token.
    """
    vocab = set([])
    count_dic = {}
    for doc in D:
        for token in doc:
            try:
                count_dic[token] += 1
                vocab.add(token)
            except:
                count_dic[token] = 0
    vocab.add('<unk>')
    return vocab


# In[7]:


class BBoWFeaturizer(object):
    def convert_document_to_feature_dictionary(self, doc, vocab):
        """
        Given a document represented as a list of tokens and the vocabulary
        as a set of tokens, compute the binary bag-of-words feature representation.
        This function should return a dictionary which maps from the name of the
        feature to the value of that feature.
        """
        feats = {}
        for token in doc:
            if token in vocab:
                feats[token] = 1
            else:
                feats["<unk>"] = 1
        return feats 


# In[8]:


class CBoWFeaturizer(object):
    def convert_document_to_feature_dictionary(self, doc, vocab):
        """
        Given a document represented as a list of tokens and the vocabulary
        as a set of tokens, compute the count bag-of-words feature representation.
        This function should return a dictionary which maps from the name of the
        feature to the value of that feature.
        """
        feats = {}
        for token in doc:
            if token in vocab:
                try:
                    feats[token] += 1
                except:
                    feats[token] = 1
            else:
                try:
                    feats["<unk>"] += 1
                except:
                    feats["<unk>"] = 1                    
        return feats    


# In[9]:


def compute_idf(D, vocab):
    """
    Given a list of documents D and the vocabulary as a set of tokens,
    where each document is represented as a list of tokens, return the IDF scores
    for every token in the vocab. The IDFs should be represented as a dictionary that
    maps from the token to the IDF value. If a token is not present in the
    vocab, it should be mapped to "<unk>".
    """
    doc_counts = {}
    for doc in D:
        feats = BBoWFeaturizer().convert_document_to_feature_dictionary(doc, vocab)
        for f in feats:
            try:
                doc_counts[f] += 1
            except:
                doc_counts[f] = 1
    idf_dic = {}
    len_D = float(len(D))
    for token in vocab:
        idf = np.log((len_D / doc_counts[token]))
        idf_dic[token] = idf
    return idf_dic
    
class TFIDFFeaturizer(object):
    def __init__(self, idf):
        """The idf scores computed via `compute_idf`."""
        self.idf = idf
    
    def convert_document_to_feature_dictionary(self, doc, vocab):
        """
        Given a document represented as a list of tokens and
        the vocabulary as a set of tokens, compute
        the TF-IDF feature representation. This function
        should return a dictionary which maps from the name of the
        feature to the value of that feature.
        """
        feats = {}
        counts = CBoWFeaturizer().convert_document_to_feature_dictionary(doc, vocab)
        for token in counts.keys():
            feats[token] = counts[token] * self.idf[token]
        return feats      


# In[10]:


# You should not need to edit this cell
def load_dataset(file_path):
    D = []
    y = []
    with open(file_path, 'r') as f:
        for line in f:
            instance = json.loads(line)
            D.append(instance['document'])
            y.append(instance['label'])
    return D, y

def convert_to_features(D, featurizer, vocab):
    X = []
    for doc in D:
        X.append(featurizer.convert_document_to_feature_dictionary(doc, vocab))
    return X


# In[11]:


def train_naive_bayes(X, y, k, vocab):
    """
    Computes the statistics for the Naive Bayes classifier.
    X is a list of feature representations, where each representation
    is a dictionary that maps from the feature name to the value.
    y is a list of integers that represent the labels.
    k is a float which is the smoothing parameters.
    vocab is the set of vocabulary tokens.
    
    Returns two values:
        p_y: A dictionary from the label to the corresponding p(y) score
        p_v_y: A nested dictionary where the outer dictionary's key is
            the label and the innner dictionary maps from a feature
            to the probability p(v|y). For example, `p_v_y[1]["hello"]`
            should be p(v="hello"|y=1).
    """
    p_y = {}
    p_v_y = {}
    y = np.array(y)
    labels = set(y)
    total = float(len(y))
    for label in labels:
        label_idx = np.where(y == label)[0]
        p_y[label] = len(label_idx) / total    
        label_token_sum_dic = {}
        label_vocab_total = 0                       
        for idx in label_idx:
            doc = X[idx]
            for token in doc.keys():
                label_vocab_total += doc[token]
                try:
                    label_token_sum_dic[token] += doc[token]
                except:
                    label_token_sum_dic[token] = doc[token]
        p_v_y[label] = {}
        vocab_len = len(vocab)
        for token in vocab:
            try:
                p_v_y[label][token] = (k + label_token_sum_dic[token]) / float(k * vocab_len + label_vocab_total)
            except:
                p_v_y[label][token] = k / float(k * vocab_len + label_vocab_total)
    return p_y, p_v_y  


# In[12]:


def predict_naive_bayes(D, p_y, p_v_y):
    """
    Runs the prediction rule for Naive Bayes. D is a list of documents,
    where each document is a list of tokens.
    p_y and p_v_y are output from `train_naive_bayes`.
    
    Note that any token which is not in p_v_y should be mapped to
    "<unk>". Further, the input dictionaries are probabilities. You
    should convert them to log-probabilities while you compute
    the Naive Bayes prediction rule to prevent underflow errors.
    
    Returns two values:
        predictions: A list of integer labels, one for each document,
            that is the predicted label for each instance.
        confidences: A list of floats, one for each document, that is
            p(y|d) for the corresponding label that is returned.
    """
    labels = list(p_y.keys())
    predictions = []
    confidences = []
    for doc in D:
        label_LLs = []
        for label in labels:
            label_prob = np.log(p_y[label])
            for token in doc:
                try:
                    label_prob += np.log(p_v_y[label][token])
                except:
                    label_prob += np.log(p_v_y[label]['<unk>'])
            label_LLs.append(label_prob)
        label_LLs = np.array(label_LLs)
        log_prob_data = np.logaddexp.reduce(label_LLs)
        pred_idx = np.argmax(label_LLs)
        predictions.append(labels[pred_idx])
        log_conf = label_LLs[pred_idx] - log_prob_data
        #confidences.append(np.exp(log_conf))
        confs=[]
        for i in range(len(label_LLs)):
            log_conf = label_LLs[i] - log_prob_data
            pred_prob = np.exp(log_conf)
            confs.append(pred_prob)
        #confidences.append(np.exp(log_conf))
        confidences.append(confs)
    return predictions, confidences, labels

# In[13]:


def train_semi_supervised(X_sup, y_sup, D_unsup, X_unsup, D_valid, y_valid, k, vocab, mode):
    """
    Trains the Naive Bayes classifier using the semi-supervised algorithm.
    
    X_sup (Sx): A list of the featurized supervised documents.
    y_sup (Sy): A list of the corresponding supervised labels.
    D_unsup (Ud): The unsupervised documents.
    X_unsup (Ux): The unsupervised document representations.
    D_valid: The validation documents.
    y_valid: The validation labels.
    k: The smoothing parameter for Naive Bayes.
    vocab: The vocabulary as a set of tokens.
    mode: either "threshold" or "top-k", depending on which selection
        algorithm should be used.
    
    Returns the final p_y and p_v_y (see `train_naive_bayes`) after the
    algorithm terminates.    
    """
    stop = False
    counter = 0
    p_y_list = []
    p_v_y_list = []
    
    while stop == False:
        p_y, p_v_y = train_naive_bayes(X_sup, y_sup, k, vocab)
        p_y_list.append(p_y)
        p_v_y_list.append(p_v_y)        
        if counter == 0:
            val_preds_0, val_confs_0 = predict_naive_bayes(D_valid, p_y, p_v_y)
            val_acc_0 = accuracy_score(y_valid, val_preds_0)
            print('starting validation accuracy: %s'%val_acc_0)
            trash = []
        counter += 1
        #U_y, P_y  returned
        preds, confs = predict_naive_bayes(D_unsup, p_y_list[-1], p_v_y_list[-1])
        #stuff to add to the supervised set
        if mode == "threshold":
            print('running threshold mode at confidence above 0.98...')
            conf_thresh = 0.98
            add_to_sup_idxes = np.where(np.array(confs) > conf_thresh)[0]
        elif mode == "top-k":
            print('running top-K mode with K=10,000...')
            K = 10000
            #reverse sort indexes, top K 
            add_to_sup_idxes = np.argsort(np.array(confs))[::-1][:K] 
        else:
            raise(RuntimeError('unrecognized mode (%s). Use "threshold" or "top-k"'%mode))        
        
        if len(add_to_sup_idxes) == 0:
            stop = True
        else:
            #Add_to, Remove_from
            print('for step %s will add %s to supervised list'%(counter, len(add_to_sup_idxes)))
            print('\t\tStarting Sup length %s; Unsup length %s'%(len(X_sup), len(X_unsup)))   
            X_sup, X_unsup = add_and_remove_indexes(X_sup, X_unsup, add_to_sup_idxes)
            y_sup, preds = add_and_remove_indexes(y_sup, preds, add_to_sup_idxes)
            trash, D_unsup = add_and_remove_indexes(trash, D_unsup, add_to_sup_idxes)
            print('\t\tResulting Sup length %s; Unsup length %s'%(len(X_sup), len(X_unsup)))   
    print('\twent through %s iterations before stopping'%counter)
    val_preds, val_confs = predict_naive_bayes(D_valid, p_y_list[-1], p_v_y_list[-1])
    val_acc_fin = accuracy_score(y_valid, val_preds)
    print('final validation accuracy: %s'%val_acc_fin)
    return p_y_list[-1], p_v_y_list[-1] 


# In[14]:


def add_and_remove_indexes(list_to_add_to, list_to_delete_from, indexes_from_delete_list):
    
    '''
    Takes two lists and a a list of valid indexes that correspond to
    the items that should be moved from the second list to the first.
    Items are deleted from the second and appended to the end of the first. 
    Returns both lists 
    '''
    a = list_to_add_to
    b = list_to_delete_from
    for index in sorted(indexes_from_delete_list, reverse = True):
        a.append(list_to_delete_from[index]) 
        #print('removing index %s'%index)
        del b[index]
        #print('adding list len: %s ; del list len: %s'%(len(list_to_add_to), len(list_to_delete_from)))
    return a, b


# In[15]:


def run_semi_supervised(k = 0.1, mode = 'top-k', starting_size = 50, data_path = './', featurizer = CBoWFeaturizer()):
    
    '''
    Function that splits training data based on starting_size and
    trains semi-supervised NB based on given k, mode, and featurizer method
    '''
    random.seed(111) 
    D_train, y_train = load_dataset(data_path + 'data/train.jsonl')
    D_valid, y_valid = load_dataset(data_path + 'data/valid.jsonl')    
    vocab = get_vocabulary(D_train)
    X_train = convert_to_features(D_train, featurizer, vocab)
    
    rand_indexes = random.sample(range(0, len(y_train)), starting_size)
    print('first 10 indexes for labeled data are: %s'%(rand_indexes[:10]))
    D_sup = []
    y_sup = []
    X_sup = []
    #Add_to, Remove_From
    D_sup, D_train = add_and_remove_indexes(D_sup, D_train, rand_indexes)
    X_sup, X_train = add_and_remove_indexes(X_sup, X_train, rand_indexes)
    y_sup, y_train = add_and_remove_indexes(y_sup, y_train, rand_indexes)
    print('sup data size: %s and unsup size: %s'%(len(D_sup), len(D_train)))
    p_y_FINAL, p_v_y_FINAL = train_semi_supervised(X_sup, y_sup, D_train, X_train, D_valid, y_valid, k, vocab, mode)
    D_test, y_test = load_dataset(data_path + 'data/test.jsonl')
    test_pred, test_conf = predict_naive_bayes(D_test, p_y_FINAL, p_v_y_FINAL)
    test_acc = accuracy_score(y_test, test_pred)
    print("FINAL TEST ACCURACY: %s"%test_acc)
    


# # Part 1: NB Experiment

# In[16]:


# Variables that are named D_* are lists of documents where each
# document is a list of tokens. y_* is a list of integer class labels.
# X_* is a list of the feature dictionaries for each document.
# D_train, y_train = load_dataset('data/train.jsonl')
# D_valid, y_valid = load_dataset('data/valid.jsonl')
# D_test, y_test = load_dataset('data/test.jsonl')

csv_path='annotated_names_original_wNamePrisms4.csv'
the_ngram = 3
k = 5
D_train, D_test, D_valid, y_train, y_test, y_valid, name_train, name_test, name_valid = parse_filter_csv(csv_path,delim=',',fn_idx=1,ln_idx=3,label_idx=12,num_cols=13, split_data=True, ngram = the_ngram)

vocab = get_vocabulary(D_train)
print(len(vocab))


# ### Extract features for the three methods

# In[17]:



vocab = get_vocabulary(D_train)
print(len(vocab))

print('extracting BBoW')
featurizer = BBoWFeaturizer()
X_train_BBoW = convert_to_features(D_train, featurizer, vocab)
print('extracting CBoW')
featurizer = CBoWFeaturizer()
X_train_CBoW = convert_to_features(D_train, featurizer, vocab)
print('extracting TFIDF')
featurizer = TFIDFFeaturizer(compute_idf(D_train, vocab))
X_train_TFIDF = convert_to_features(D_train, featurizer, vocab)



# In[18]:
author_name, author_doc, author_label = parse_filter_csv('authors.tsv',delim='\t',fn_idx=1,ln_idx=3)
#print(author_name)
keynote_name, keynote_doc, keynote_label = parse_filter_csv('ISMB_Keynotes.txt',delim='\t',fn_idx=2,ln_idx=3)
#print(keynote_name)



#ks = [0.001, 0.01, 0.1, 1.0, 10.0]
ks = [1, 5, 10]

BBoW_p_ys = []
BBoW_p_v_ys = []
CBoW_p_ys = []
CBoW_p_v_ys = []
TFIDF_p_ys = []
TFIDF_p_v_ys = []
for k in ks:
    print('training NB on BBoW with k=%s...'%k)
    BBoW_p_y, BBoW_p_v_y = train_naive_bayes(X_train_BBoW, y_train, k, vocab)
    BBoW_p_ys.append(BBoW_p_y)
    BBoW_p_v_ys.append(BBoW_p_v_y)
    print('training NB on CBoW with k=%s...'%k)
    CBoW_p_y, CBoW_p_v_y = train_naive_bayes(X_train_CBoW, y_train, k, vocab)
    CBoW_p_ys.append(CBoW_p_y)
    CBoW_p_v_ys.append(CBoW_p_v_y)
    print('training NB on TFIDF with k=%s...'%k)
    TFIDF_p_y, TFIDF_p_v_y = train_naive_bayes(X_train_TFIDF, y_train, k, vocab)
    TFIDF_p_ys.append(TFIDF_p_y)
    TFIDF_p_v_ys.append(TFIDF_p_v_y)


# ### Get training and validation accuracies for 3 feature methods for all ks

# In[19]:


BBoW_val_accs = []
BBoW_train_accs = []

CBoW_val_accs = []
CBoW_train_accs = []

TFIDF_val_accs = []
TFIDF_train_accs = []

# In[21]:

# ### Print validation accuracy at best value of k for each

# In[22]:
pred, conf, labels = predict_naive_bayes(author_doc, BBoW_p_ys[0], BBoW_p_v_ys[0])
confidences=pd.DataFrame(conf,columns=labels)
confidences.to_csv("authors_TFIDF_distribution.csv",sep='\t')


pred, conf, labels = predict_naive_bayes(keynote_doc, BBoW_p_ys[0], BBoW_p_v_ys[0])
confidences=pd.DataFrame(conf,columns=labels)
confidences.to_csv("keynotes_TFIDF_distribution.csv",sep='\t')


#pred, conf, labels = predict_naive_bayes(D_test, BBoW_p_ys[4], BBoW_p_v_ys[4])
#test_acc_bbow = accuracy_score(y_test, pred)
#pred, conf = predict_naive_bayes(D_test, CBoW_p_ys[4], CBoW_p_v_ys[4])
#test_acc_cbow = accuracy_score(y_test, pred)
#pred, conf = predict_naive_bayes(D_test, TFIDF_p_ys[4], TFIDF_p_v_ys[4])
#test_acc_tfidf = accuracy_score(y_test, pred)
#print(test_acc_bbow, test_acc_cbow, test_acc_tfidf)

