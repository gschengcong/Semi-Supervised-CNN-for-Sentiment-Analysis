import collections
import numpy as np
import re
import tensorflow as tf

print("test")

vocabulary_size = 20000
data_index = 0

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

    features = np.ndarray(shape=(len(samples), vocabulary_size), dtype=np.int32)
    labels = np.ndarray(shape=(len(samples), 2), dtype=np.int32)

    for i in range(len(samples)):
        case = samples[i]
        label = int(case[0])
        labels[i][label] = 1
        labels[i][(label+1)%2] = 0
        x = case[2:].split()
        for j in range(len(x)):
            if x[j] in dictionary:
                features[i][dictionary[x[j]]] = 1
            else:
                features[i][0] = 1
    return features, labels

training_X, training_Y = genrate_data_matrix('combined.txt')
# print training_X, training_Y

def generate_batch(batch_size):
    global data_index
    if data_index >= len(training_X):
    	data_index -= len(training_X)
    start = data_index
    end = data_index + batch_size 
    data_index += batch_size

    return training_X[start : end], training_Y[start : end]

dev_data, dev_lables = genrate_data_matrix("dev_processed_data.txt")
test_data, test_lables = genrate_data_matrix("test_processed_data.txt")


sess = tf.InteractiveSession()

x = tf.placeholder("float", shape=[None, vocabulary_size])
y_ = tf.placeholder("float", shape=[None, 2])

# W = tf.Variable(tf.zeros([vocabulary_size,128]))
W = tf.Variable(tf.truncated_normal(shape = [vocabulary_size, 128], stddev=0.1))
b = tf.Variable(tf.constant(0.1, shape=[128]))

# W1 = tf.Variable(tf.zeros([128, 2]))
W1 = tf.Variable(tf.truncated_normal(shape = [128, 2], stddev=0.1))
b1 = tf.Variable(tf.constant(0.1, shape=[2]))

sess.run(tf.initialize_all_variables())

h1 = tf.nn.relu(tf.matmul(x,W) + b)
y = tf.nn.softmax(tf.matmul(h1, W1) + b1)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

for i in range(20000):
    points, lables = generate_batch(50)
  # print(lables, data_index)
    train_step.run(feed_dict={x: points, y_: lables})
    if(i%100 == 0):
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        points, lables = generate_batch(19206)
        print("iteration ", i, accuracy.eval(feed_dict={x: points, y_: lables}))
        print("dev data accuracy: ", accuracy.eval(feed_dict={x: dev_data, y_: dev_lables}))

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("dev data accuracy: ", accuracy.eval(feed_dict={x: dev_data, y_: dev_lables}))
print("test data accuray: ", accuracy.eval(feed_dict={x: test_data, y_: test_lables}))
