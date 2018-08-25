import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer, fully_connected
import numpy as np
import os

tf.set_random_seed(10)

class Fairseq:
    def __init__(self, config):
        self.config = config
        self.n_encoder = config['nlayers_encoder']
        self.n_decoder = config['nlayers_decoder']
        self.embeded_dimesion = config['embeded_dimesion']
        self.vocubulary_size = config['vocabulary_size']
        self.hidden_size  = config['hidden_size']
        self.learning_rate = config['learning_rate']
        self.checkpoint_dir = config['checkpoint_dir']
        self.checkpoint_step = config['checkpoint_step']
        self.total_steps = config['total_steps']

    @staticmethod
    def glu(input):
        output = tf.multiply(input[:, :, :tf.shape(input)[2]//2],
                             tf.sigmoid(input[:, :, tf.shape(input)[2]//2:]))
        return output

    def encoder_block(self):
        input = output = self.encoder_embedded
        for i in range(self.n_encoder):
            conv = tf.layers.conv1d(input,
                                    filters = self.hidden_size,
                                    kernel_size=3,
                                    padding="same")

            glu_output = self.glu(conv)
            residual = glu_output + self.encoder_embedded
            input = output = residual
        return output

    def decoder_block(self):
        input = output = self.decoder_embedded
        for i in range(self.n_decoder):
            conv = tf.layers.conv1d(input,
                                    filters=self.hidden_size,
                                    kernel_size=3,
                                    padding="same")
            d = fully_connected(conv,
                                num_outputs=self.embeded_dimesion,
                                activation_fn=None) + self.decoder_embedded

            attention = tf.matmul(d, self.encoder_output, transpose_b=True)
            attention = tf.nn.softmax(attention)
            conditional = tf.matmul(attention, self.encoder_output + self.encoder_embedded)
            output = fully_connected(conditional,
                                 num_outputs=self.embeded_dimesion,
                                 activation_fn=None)
            input = output
        return output




    def build_model(self):
        #input is the size of (N, m)

        with tf.variable_scope("encoder"):
            self.encoder_input = tf.placeholder(name = "encoder_input",
                                                shape = (None, None),
                                                dtype = tf.int32)

            encoder_embeddings = tf.get_variable(name = "encoder_embeddings",
                                             initializer=xavier_initializer(),
                                             shape = [self.vocubulary_size, self.embeded_dimesion])

            self.encoder_embedded = tf.nn.embedding_lookup(params = encoder_embeddings,
                                                   ids = self.encoder_input)

            self.encoder_output = self.encoder_block()

        with tf.variable_scope("decoder"):
            self.decoder_input = tf.placeholder(name = "decoder_input",
                                           shape = (None, None),
                                           dtype = tf.int32)
            decoder_embeddings = tf.get_variable(name="encoder_embeddings",
                                                 initializer=xavier_initializer(),
                                                 shape=[self.vocubulary_size, self.embeded_dimesion])
            self.decoder_embedded = tf.nn.embedding_lookup(params = decoder_embeddings,
                                                   ids = self.decoder_input)
            self.output = self.decoder_block()

            self.logits = fully_connected(self.output,
                                     num_outputs=self.vocubulary_size,
                                     activation_fn=None)
            self.pred = tf.nn.softmax(self.logits)


    def init_loss(self):
        with tf.variable_scope("cross_entropy_sequence_loss"):
            logits = tf.reshape(self.logits, [tf.shape(self.logits)[0] * tf.shape(self.logits)[1],
                                         self.vocubulary_size])

            labels = self.decoder_input[:,1:]
            labels = tf.reshape(labels, [-1, ])
            loss_mask = labels > 0
            logits = tf.boolean_mask(logits, loss_mask)
            labels = tf.boolean_mask(labels, loss_mask)
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                                  logits=logits)
            #loss = tf.reduce_mean(loss)
            #tf.summary.scalar('softmax_loss', loss)
            return loss


    def train(self, X, Y):
        loss = self.init_loss()
        train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)
        loss_val  = tf.reduce_mean(loss)
        
        tf.summary.scalar('softmax_loss', loss_val)
        saver = tf.train.Saver(tf.trainable_variables())
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(self.total_steps):
                loss__, train_op__, loss_val__ = sess.run([loss, train_op, loss_val], feed_dict={self.encoder_input: X,
                                                     self.decoder_input: Y})
                if i % self.checkpoint_step == 0:
                    print("Loss: ", loss_val__)
                    saver.save(sess, self.checkpoint_dir + 'fairseq_step', global_step=i)



    def unit_test(self):
        self.build_model()
        test_matrix = np.ones((1000, 10))
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        sess.run([self.pred], feed_dict={self.encoder_input:test_matrix,
                                    self.decoder_input: test_matrix})
        self.train(test_matrix, test_matrix)


if __name__ == "__main__":
    config = {"nlayers_encoder": 1,
              'nlayers_decoder': 1,
              'vocabulary_size': 10,
              'hidden_size': 2*256,
              'embeded_dimesion': 256,
              'learning_rate': 0.001,
              'checkpoint_dir': 'checkpoints/',
              'checkpoint_step': 10,
              'total_steps': 100}

    if not os.path.exists(config['checkpoint_dir']):
        os.mkdir(config['checkpoint_dir'])

    fairseq = Fairseq(config)
    fairseq.unit_test()