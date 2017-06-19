import tensorflow as tf
import csv
import os

''' ---------- LOADING SESSION ------------

training data
input:
    'total_x.csv': x_data
    'total_y.csv': y_data
output:
    'w.csv': coef_w
    'b.csv': coef_b
'''

learning_rate = 0.0001
config = {
    'learning_rate':    0.0001,
    'x_data_fn':    'total_x.csv',
    'y_data_fn':    'total_y.csv',
    'restore': True,
    'save_path': './restore/model_bin.ckpt'
}
directions = {
    'front':   [1, 0, 0, 0, 0],
    'front 3 quarter':  [0, 1, 0, 0, 0],
    'rear':       [0, 0, 1, 0, 1],
    'rear 3 quarter':   [0, 0, 0, 1, 0],
    'side':       [0, 0, 0, 0, 1]
}

x_data = []
y_data = []


def xaver_init(n_inputs, n_outputs, uniform=True):
    if uniform:
        init_range = tf.sqrt(6.0 / (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)

    else:
        stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)


def acc(d1, d2):
    cnt = 0
    for i in range(d1.__len__()):
        if d1[i] == d2[i]:
            cnt += 1

    return float(cnt)/d1.__len__()


def sel_max(data):
    ret_ind = []
    for i in range(data.__len__()):
        if data[i][0] == 1:
            ret_ind.append(0)
        else:
            ret_ind.append(1)

    return ret_ind


""" Loading training data from csv files """
print('[Step 1] Loading training data ...')
with open(config['x_data_fn']) as fp:
    csv_reader = csv.reader(fp, delimiter=',')
    for row in csv_reader:
        x_data.append([float(str_val) for str_val in row])

with open(config['y_data_fn']) as fp:
    csv_reader = csv.reader(fp, delimiter=',')
    for row in csv_reader:
        y_data.append(directions[row[0]])


""" Placeholder """
print('[Step 2] Placeholder')
x = tf.placeholder('float', [None, 2048])  # len(feature) = 2048
y = tf.placeholder('float', [None, 5])  # len(Directions) = 5 : classes

W1 = tf.get_variable('W1', shape=[2048, 5], initializer=xaver_init(2048, 5))
b1 = tf.Variable(tf.zeros([5]))
activation = tf.add(tf.matmul(x, W1), b1)
t1 = tf.nn.softmax(activation)


""" Minimize error using cross entropy """
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=activation, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)  # Gradient Descent


""" Initializing the variables """
print('[Step 3] Initializing the variables.')
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)
saver = tf.train.Saver()

if config['restore']:
    print('Loading the last learning Session.')
    saver.restore(sess, config['save_path'])


""" Training cycle """
print('[Step 4] Training...')
for step in range(20000):
    sess.run(optimizer, feed_dict={x: x_data, y: y_data})
    if step % 10 == 0:
        ret = sess.run(t1, feed_dict={x: x_data})
        ret1 = sel_max(ret)
        acc1 = acc(ret1, sel_max(y_data))*100

        print('    ', step, sess.run(cost, feed_dict={x: x_data, y: y_data}), acc1)

        saver.save(sess, config['save_path'])

print('Optimization Finished!')
