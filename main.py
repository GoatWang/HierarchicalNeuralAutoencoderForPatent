from __future__ import division
from __future__ import print_function

from models.hier_autoencoder import HierarchicalAutoencoder as HAE
from paper_data_utils import create_vocabulary
from paper_data_utils import initialize_vocabulary
from paper_data_utils import data_iterator

import tensorflow as tf
import numpy as np
from datetime import datetime
import os

data_dir = "training_data"

# data_path = "paper_data/train_target_permute_segment.txt"
# sample_data_path = "paper_data/sample_text.txt"
# debug_data_path = "paper_data/debug_text.txt"
patent_data_path = os.path.join(data_dir, "training_text_top1.txt")


training_data_path = patent_data_path
vocab_path = os.path.join(data_dir, "vocab")

size = 16

def main():
    stime = datetime.now()
    with tf.Session() as sess:
        #max_sent_len, max_doc_len = create_vocabulary("data/vocab",
        #                                              "data/sample.txt", 1000)
        #vocab, rev_vocab = initialize_vocabulary("data/vocab")
        
        max_sent_len, max_doc_len = create_vocabulary(vocab_path, training_data_path, 50)
        vocab, _ = initialize_vocabulary(vocab_path)
        print("vocabulary have been initialized!")
        print("max_sent_len=" + str(max_sent_len))
        print("max_doc_len="  + str(max_doc_len))
        # print("len(vocab)=" + str(len(vocab)))
        # print("vocab=" + str(vocab))

        # model = HAE(vocab, max_sent_len, max_doc_len, size=size, batch_size=2)
        # print("model have been build!")
        # print(model)

        # sess.run(tf.initialize_all_variables())
        # print("tf variables have been initialized!")

        # model.train(sess, training_data_path, data_iterator, iterations=500, save_iters=25)
    print(datetime.now() - stime)

def sample():
    with tf.Session() as sess:
        max_sent_len, max_doc_len = 74, 16
        vocab, rev_vocab = initialize_vocabulary(vocab_path)

        model = HAE(vocab, max_sent_len, max_doc_len, size=size)
        model.load(sess, "checkpoint")

        for i,(x,y) in enumerate(data_iterator(training_data_path, vocab,
                                               max_sent_len, max_doc_len)):
            x_hat = model.sample(sess, x, rev_vocab)
            print(x)
            print(x_hat)
            break

if __name__ == '__main__':
    main()
