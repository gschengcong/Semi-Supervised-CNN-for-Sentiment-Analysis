import collections
import numpy as np
import re
import tensorflow as tf
import time

print("test")

vocabulary_size = 12000
data_index = 0
max_sentence_length = 66

def read_data(filename):
    f = open(filename, 'r')
    data = f.read()
    data = re.sub(r'\n[01]', '', data)
    words = data.split()
    return words
words = read_data("combined.txt")

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
    print(unk_count)
    return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words)

def genrate_data_matrix(filename):
    f = open(filename, 'r')
    data = f.read()
    samples = data.split("\n")
    features = np.ndarray(shape=(len(samples), max_sentence_length, vocabulary_size), dtype=np.int32)
    labels = np.ndarray(shape=(len(samples), 2), dtype=np.int32)
    for i in range(len(samples)):
        case = samples[i]
        label = int(case[0])
        labels[i][label] = 1
        labels[i][(label+1)%2] = 0
        x = case[2:].split()
        for j in range(len(x)):
            if x[j] in dictionary:
                features[i][j][dictionary[x[j]]] = 1
            else:
                features[i][j][0] = 1
    return features, labels


def generate_batch(batch_size, filename):
    global data_index
    f = open(filename, 'r')
    data = f.read()
    samples = data.split("\n")
    features = np.ndarray(shape=(batch_size, max_sentence_length, vocabulary_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 2), dtype=np.int32)
    if data_index + batch_size >= len(samples):
        data_index -= len(samples) 
        data_index += batch_size
        data_index /= 2
    for i in range(batch_size):
        case = samples[i + data_index]
        label = int(case[0])
        labels[i][label] = 1
        labels[i][(label+1)%2] = 0
        x = case[2:].split()
        for j in range(len(x)):
	    if x[j] in dictionary:
	        features[i][j][dictionary[x[j]]] = 1
	    else:
	        features[i][j][0] = 1
    data_index += batch_size	
    return features, labels

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, max_sentence_length, vocabulary_size])
y_ = tf.placeholder(tf.float32, shape=[None, 2])


W_conv1 = weight_variable([2, 1, vocabulary_size, 32])

b_conv1 = bias_variable([32])

x_sentence = tf.reshape(x, [-1, max_sentence_length, 1, vocabulary_size])

h_conv1 = tf.nn.relu(conv2d(x_sentence, W_conv1) + b_conv1)
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='SAME')

W_fc = weight_variable([max_sentence_length*32/2, 2])
b_fc = weight_variable([2])

h_flat = tf.reshape(h_pool1, [-1, max_sentence_length*32/2])
y_conv =tf.nn.softmax(tf.matmul(h_flat, W_fc) + b_fc)

sess.run(tf.initialize_all_variables())


cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())
for i in range(200000):
  start_time = time.time()
  batch, labels = generate_batch(50, "combined.txt")
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={x:batch, y_: labels})
    duration = time.time() - start_time
    print('iteration %d: training data accuracy = %.2f (%.3f sec)' % (i, train_accuracy , duration))
  train_step.run(feed_dict={x: batch, y_: labels})
  duration = time.time() - start_time
  # print('iteration %d: (%.3f sec)' % (i, duration))
  if i% 500 == 0:
    dev_data, dev_lables = genrate_data_matrix("dev_processed_data.txt")
    dev_accuracy = accuracy.eval(feed_dict={x:dev_data, y_: dev_lables})
    duration = time.time() - start_time - duration
    print('iteration %d: dev data accuracy = %.2f (%.3f sec)' % (i, dev_accuracy , duration))
#    del dev_data
#    del dev_lables

#    test_data, test_lables = genrate_data_matrix("test_processed_data.txt")
#    test_accuracy = accuracy.eval(feed_dict={x:test_data, y_: test_lables})
#    print("iteration %d, test accuracy %g"%(i, test_accuracy))

dev_data, dev_lables = genrate_data_matrix("dev_processed_data.txt")
dev_accuracy = accuracy.eval(feed_dict={x:dev_data, y_: dev_lables})
print("step %d, dev accuracy %g"%(i, dev_accuracy))

test_data, test_lables = genrate_data_matrix("test_processed_data.txt")
test_accuracy = accuracy.eval(feed_dict={x:test_data, y_: test_lables})
print("step %d, test accuracy %g"%(i, test_accuracy))

