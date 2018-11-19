'''
blog_post_generator.py

by Dominic Reichl, @domreichl

November 2018

NLP model that uses a LSTM/GRU-RNN to learn to write blog posts

Overview:
    1. Preprocess data
    2. Build neural network
    3. Start training
    4. Generate text
'''

import nltk, time, sys
import numpy as np
import tensorflow as tf

'''
---------------------------
1. Preprocess Data
---------------------------
'''

# get entire blog content in one string
with open("blog_posts_raw.txt") as file:
    text = file.read()
    print('Text contains', len(text), 'characters.')

# set sentence structure
unknown_token = "<unk>"
sentence_start_token = "START"
sentence_end_token = "END"
    
# parse sentences from text and tokenize them into words
sentences = ["%s %s %s" % (sentence_start_token, sent, sentence_end_token) for sent in (nltk.sent_tokenize(text))]
tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
print("Parsed %d sentences." % len(sentences))

# count word frequencies
word_freq = nltk.FreqDist((word for sent in tokenized_sentences for word in sent))
print("Found %d unique words tokens." % len(word_freq.items()))

# build vocabulary with most common words
vocab_size = 5990
vocab = word_freq.most_common(vocab_size-1)
print("Using vocabulary size %d." % vocab_size)
print("The least frequent word is '%s', which appears %d times." % (vocab[-1][0], vocab[-1][1]))

# build index_to_word and word_to_index vectors
index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)
word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

# replace all words not in vocabulary with the unknown token
for i, sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

print("\nExample sentence: '%s'" % sentences[0])
print("\nExample sentence after pre-processing: '%s'" % tokenized_sentences[0])

# create training data
X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

# print training data example
print("\nTraining example x:\n%s\n%s" % (" ".join([index_to_word[x] for x in X_train[1200]]), X_train[1200]))
print("\nTraining example y:\n%s\n%s" % (" ".join([index_to_word[x] for x in y_train[1200]]), y_train[1200]))

'''
---------------------------
2. Build Neural Network
---------------------------
'''

class LSTM():
    def __init__(self, n_in, n_out, n_state, GRU=False):
        ''' builds tensorflow graph with LSTM forward passes, backpropagation, and adaptive learning rates '''
        self.graph = tf.Graph() # construct computation graph
        
        with self.graph.as_default():
            # memory cell and activation
            self.c = tf.Variable(tf.zeros((n_state,1)))
            self.a = tf.Variable(tf.zeros((n_state,1)))

            # weights and biases
            if not GRU:
                self.X = tf.Variable(tf.random_uniform((4,n_state,n_in), minval=-np.sqrt(1./n_in), maxval=np.sqrt(1./n_in)))
                self.W = tf.Variable(tf.random_uniform((4,n_state,n_state), minval=-np.sqrt(1./n_state), maxval=np.sqrt(1./n_state)))
                self.b = tf.Variable(tf.ones((4,n_state,1)))
            else:
                self.X = tf.Variable(tf.random_uniform((3,n_state,n_in), minval=-np.sqrt(1./n_in), maxval=np.sqrt(1./n_in)))
                self.W = tf.Variable(tf.random_uniform((3,n_state,n_state), minval=-np.sqrt(1./n_state), maxval=np.sqrt(1./n_state)))
                self.b = tf.Variable(tf.ones((3,n_state,1)))
            self.Wy = tf.Variable(tf.random_uniform((n_out,n_state), minval=-np.sqrt(1./n_state), maxval=np.sqrt(1./n_state)))
            self.by = tf.Variable(tf.ones((n_out,1)))

            # placeholders for input, output, learning rate, decay rate
            x_words = tf.placeholder(tf.int32, [None])
            y_words = tf.placeholder(tf.int32, [None])
            learn_r = tf.placeholder(tf.float32)
            decay_r = tf.placeholder(tf.float32)

            # adaptive learning rates
            self.lrX = tf.Variable(tf.zeros(self.X.shape))
            self.lrW = tf.Variable(tf.zeros(self.W.shape))
            self.lrb = tf.Variable(tf.zeros(self.b.shape))
            self.lrWy = tf.Variable(tf.zeros(self.Wy.shape))
            self.lrby = tf.Variable(tf.zeros(self.by.shape))

            # initialize variables and saver
            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()

            # a single foward step
            def forward(cay, word):
                c, a, y = cay # previous cell outputs

                if not GRU:
                    # gates: input, forget, output
                    i = tf.sigmoid(tf.reshape(self.X[0,:,word], (-1,1)) + tf.matmul(self.W[0], a) + self.b[0])
                    f = tf.sigmoid(tf.reshape(self.X[1,:,word], (-1,1)) + tf.matmul(self.W[1], a) + self.b[1])
                    o = tf.sigmoid(tf.reshape(self.X[2,:,word], (-1,1)) + tf.matmul(self.W[2], a) + self.b[2])

                    # candidate for replacing c and memory update
                    c_tilde = tf.tanh(tf.reshape(self.X[3,:,word], (-1,1)) + tf.matmul(self.W[3], a) + self.b[3])
                    c = i*c_tilde + f*c

                    # new state and output
                    a = tf.tanh(c)*o
                    y = tf.matmul(self.Wy, a) + self.by

                else:
                    # gates: reset, update
                    r = tf.sigmoid(tf.reshape(self.X[0,:,word], (-1,1)) + tf.matmul(self.W[0], c) + self.b[0])
                    u = tf.sigmoid(tf.reshape(self.X[1,:,word], (-1,1)) + tf.matmul(self.W[1], c) + self.b[1])

                    # memory update
                    c_tilde = tf.tanh(tf.reshape(self.X[2,:,word], (-1,1)) + tf.matmul(self.W[2], r*c) + self.b[2])
                    c = u*c_tilde + (1-u)*c
                    a = c

                y = tf.matmul(self.Wy, a) + self.by
                return [c, a, y] # new cell outputs

            # step through input sequence, word by word
            cay_init = [self.c, self.a, tf.zeros((n_out,1))]
            results = tf.scan(forward, x_words, cay_init) 
            outputs = results[2]

            # update current cell state
            update_c = self.c.assign(results[0][-1])
            update_a = self.a.assign(results[1][-1])

            # compute errors using cross entropy
            errors = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=outputs[..., 0], labels=y_words)
            errors = tf.reduce_mean(errors)

            # compute derivatives
            dX = tf.gradients(errors, self.X)[0]
            dW = tf.gradients(errors, self.W)[0]
            db = tf.gradients(errors, self.b)[0]
            dWy = tf.gradients(errors, self.Wy)[0]
            dby = tf.gradients(errors, self.by)[0]

            # update rmsprop learning rate
            update_lrX = self.lrX.assign(decay_r * self.lrX + (1 - decay_r) * dX ** 2)
            update_lrW = self.lrW.assign(decay_r * self.lrW + (1 - decay_r) * dW ** 2)
            update_lrb = self.lrb.assign(decay_r * self.lrb + (1 - decay_r) * db ** 2)
            update_lrWy = self.lrWy.assign(decay_r * self.lrWy + (1 - decay_r) * dWy ** 2)
            update_lrby = self.lrby.assign(decay_r * self.lrby + (1 - decay_r) * dby ** 2)   
                
            # nudge weights using updated learning rates
            nudge_X = self.X.assign(self.X - learn_r*dX/tf.sqrt(self.lrX + 1e-6))
            nudge_W = self.W.assign(self.W - learn_r*dW/tf.sqrt(self.lrW + 1e-6))
            nudge_b = self.b.assign(self.b - learn_r*db/tf.sqrt(self.lrb + 1e-6))
            nudge_Wy = self.Wy.assign(self.Wy - learn_r*dWy/tf.sqrt(self.lrWy + 1e-6))   
            nudge_by = self.by.assign(self.by - learn_r*dby/tf.sqrt(self.lrby + 1e-6))

            # re-initialize cell and state
            reset_c = self.c.assign(tf.zeros((n_state,1)))
            reset_a = self.a.assign(tf.zeros((n_state,1)))

            # backpropagate through time by nudging weights based on the pair of sequences x and y
            def bptt(x, y, learning_rate):
                results = self.session.run([reset_c, reset_a, errors, update_lrX, update_lrW, update_lrb, update_lrWy, update_lrby,
                                            nudge_Wy, nudge_X, nudge_W, nudge_b, nudge_by, update_c, update_a],
                                           feed_dict={x_words: x, y_words: y, learn_r: learning_rate, decay_r: 0.9})
                return results[2]
            self.bptt = bptt

            # compute outputs without differentiation
            def predict(x):
                outputs_pred = self.session.run([outputs, update_c, update_a], feed_dict={x_words: x})[0]
                outputs_pred = tf.nn.softmax(outputs_pred[...,0])
                return outputs_pred
            self.predict = predict

    def fit(self, X_train, y_train, epochs=5, learning_rate=0.001, restore=False):
        ''' fits LSTM model '''
        indices = list(range(len(X_train)))

        with tf.Session(graph=self.graph).as_default() as self.session:
            if not restore:
                self.session.run(self.init)
            else:
                self.saver.restore(self.session, "/tmp/model.ckpt")
            for e in range(epochs):
                np.random.shuffle(indices)
                smooth_loss = 0
                t = time.time()
                print("Epoch #" + str(e+1) + " started")
                for i in range(len(X_train)):
                    x = X_train[indices[i]]
                    y = y_train[indices[i]]
                    errors = self.bptt(x, y, learning_rate)
                    smooth_loss = (errors + smooth_loss*i)/(i+1)
                    if i%2000 == 0:
                        print("Example " + str(i+1) + ", Loss " + str(smooth_loss) + ", Seconds " + str(time.time()-t))
                print("Epoch #" + str(e+1) + " completed, Loss " + str(smooth_loss) + ", Minutes " + str((time.time()-t)/60) + "\n")
                self.saver.save(self.session, "/tmp/model.ckpt")

'''
---------------------------
3. Start Training
---------------------------
'''

# build LSTM/GRU-RNN with proper dimensions
model = LSTM(n_in=vocab_size, n_out=vocab_size, n_state=128, GRU=True)

# start training
user_input = ""
while user_input != 'run' and user_input != 'restore':
    user_input = input("\nDo you want to 'run' or 'restore' the model? ")
if user_input == 'run':
    model.fit(X_train, y_train, epochs=1)
    txt = open("model_output.txt", "w")
else:
    model.fit(X_train, y_train, epochs=1, learning_rate = 0.0001, restore=True)
    txt = open("model_output.txt", "a")

'''
---------------------------
4. Generate Text
---------------------------
'''

with tf.Session():
    for i in range(10): # generate one sentence per iteration
        last_pred = sentence_start_token
        while True: # generate one word per iteration
            probs = model.predict([word_to_index[last_pred]]).eval()[-1]
            next_pred = np.random.multinomial(1, probs/np.sum(probs+1e-6))
            next_pred = index_to_word[np.argmax(next_pred)]
            if next_pred == sentence_end_token:
                if last_pred in (".", "?", "!", ",", ":", ";"):
                    print()
                else:
                    print(".")
                break
            else:
                if next_pred in (".", "?", "!", ",", ":", ";", "'"):
                    sys.stdout.write(next_pred)
                    txt.write(next_pred)
                else:
                    sys.stdout.write(" " + next_pred)
                    txt.write(" " + next_pred)
                last_pred = next_pred
        txt.write("\n")
    txt.write("\n")
    txt.close()
