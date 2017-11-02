import os

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

user_input = tf.placeholder(tf.int32, shape = [None, 1], name = "user_input")
item_input = tf.placeholder(tf.int32, shape = [None, 1], name = "item_input")
labels = tf.placeholder(tf.float32, shape = [None, 1], name = "labels")

if __name__ == '__main__':

    with tf.Session() as sess:
    #     dataset = Dataset("./Data/ml-1m")
    #     batches = shuffle(dataset, 10)
    #     print batches
        test = []
        for i in range(100):
            test += [[i],[i],[i]]
        print test
        batch = tf.train.shuffle_batch(test, batch_size = 50, capacity = 20, min_after_dequeue = 0,  num_threads = 4)
        print batch
        print "+++++++"
        print sess.run(batch[0][0])
        # print sess.run([user_input], feed_dict = {user_input: batch[0], item_input: batch[1], labels: batch[2]})

