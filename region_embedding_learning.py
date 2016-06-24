import collections
import numpy as np
import re
import tensorflow as tf
import time
import random
import math

print("test")

vocabulary_size = 50000
data_index = 0

def read_data(filename):
    f = open(filename, 'r')
    data = f.read()
    data = re.sub(r'\n[01]', '', data)
    data = data.lower()
    words = data.split()
    return words
# cannot upload the unlabled data, since it is too large.
# please go to "http://riejohnson.com/cnn_download.html" to download the CONTEXT v3.00, which contains the unlabled data. 
words = read_data("elec-25k-unlab00.txt.tok")

# print(words)


def build_dataset(words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count = unk_count + 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys())) 
    print("unknown counts: ", unk_count)
    return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words)

data_index = 3

# print data
# print('data:', [reverse_dictionary[di] for di in data[:15]])
# print("data: ", data[0:15])

# generate training data.
def generate_batch(batch_size, region_size):
    global data_index
    if data_index + region_size >= len(data):
        data_index = len(data) - data_index
    batch = np.ndarray(shape=(batch_size, vocabulary_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, vocabulary_size), dtype=np.int32)
    for i in range(batch_size):
        count = 0
        for j in range(region_size):
            batch[i][data[data_index + j]] = 1
            labels[i][data[data_index - j - 1]] = 1
            count += 1
            labels[i][data[data_index + j + region_size]] = 1
            count += 1
        data_index += region_size
    return batch, labels

# batch, labels = generate_batch(3, 2)
# print batch
# print labels

# saver = tf.train.Saver(tf.all_variables())
sess = tf.InteractiveSession()


x = tf.placeholder("float", shape=[None, vocabulary_size])
y_ = tf.placeholder("float", shape=[None, vocabulary_size])

# W = tf.Variable(tf.zeros([vocabulary_size,128]))
W = tf.Variable(tf.truncated_normal(shape = [vocabulary_size, 128], stddev=0.1))
b = tf.Variable(tf.constant(0.1, shape=[128]))

W1 = tf.Variable(tf.truncated_normal(shape = [128, vocabulary_size], stddev=0.1))
b1 = tf.Variable(tf.constant(0.1, shape=[vocabulary_size]))

sess.run(tf.initialize_all_variables())
h1 = tf.nn.relu(tf.matmul(x,W) + b)
y = tf.nn.softmax(tf.matmul(h1, W1) + b1)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

for i in range(100000):
    print "iteration ", i
    points, lables = generate_batch(512, 2)
  # print(lables, data_index)
    train_step.run(feed_dict={x: points, y_: lables})
    # save_path = saver.save(sess, "model.ckpt")
    # print("Model saved in file: %s" % save_path)
