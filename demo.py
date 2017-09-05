# -*- coding: utf-8 -*-
import codecs
import re
import jieba
import nltk
import itertools
from tensorflow.python.platform import gfile
import tensorflow as tf
import numpy as np
import random
import time
import math
import sys
import logging
import os

_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"

_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

vocabulary_size = 30000
vocabulary_path = 'vocab/vocab-list'

_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]

class Seq2SeqModel(object):
    def __init__(self, 
                 usize, 
                 vocab_size, 
                 num_layers, 
                 buckets, 
                 max_gradient_norm, 
                 batch_size, 
                 learning_rate, 
                 learning_rate_decay_factor, 
                 use_lstm=False, 
                 forward_only=False, 
                 num_samples=512, 
                 dtype=tf.float32):
        self.buckets = buckets
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=dtype)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)
        
        # 使用sampled softmax時，需要 output projection (不清楚)
        output_projection = None
        softmax_loss_function = None
        
        if num_samples > 0 and num_samples < self.vocab_size:
            w_t = tf.get_variable("proj_w", [self.vocab_size, usize], dtype=dtype)
            w = tf.transpose(w_t)
            b = tf.get_variable("proj_b", [self.vocab_size], dtype=dtype)
            output_projection = (w, b)
            
            def sampled_loss(labels, logits):
                labels = tf.reshape(labels, [-1, 1])
                local_w_t = tf.cast(w_t, tf.float32)
                local_b = tf.cast(b, tf.float32)
                local_inputs = tf.cast(logits, tf.float32)
                return tf.cast(
                    tf.nn.sampled_softmax_loss(
                        weights=local_w_t,
                        biases=local_b,
                        labels=labels,
                        inputs=local_inputs,
                        num_sampled=num_samples,
                        num_classes=self.vocab_size), dtype)
            
            softmax_loss_function = sampled_loss
        
        def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
            # 建立 multi-layer cell 給 RNN 用
            def single_cell():
                return tf.contrib.rnn.GRUCell(usize)
            if use_lstm:
                def single_cell():
                    return tf.contrib.rnn.BasicLSTMCell(usize)
            cell = single_cell()

            if num_layers > 1:
                cell = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(num_layers)])
            return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                encoder_inputs,
                decoder_inputs,
                cell,
                num_encoder_symbols=vocab_size,
                num_decoder_symbols=vocab_size,
                embedding_size=usize,
                output_projection=output_projection,
                feed_previous=do_decode,
                dtype=dtype)
        
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []
        # 最後一個bucket的 size 最大，取 source
        for i in xrange(buckets[-1][0]):
            self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="encoder{0}".format(i)))
        
        for i in xrange(buckets[-1][1]+1):
            self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="decoder{0}".format(i)))
            self.target_weights.append(tf.placeholder(dtype, shape=[None], name="weight{0}".format(i)))
        
        # 往右移一位，估計是要保留給 "GO" 符號
        targets = [self.decoder_inputs[i+1] for i in xrange(len(self.decoder_inputs)-1)]
        
        if forward_only:
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                self.encoder_inputs, self.decoder_inputs, targets,
                self.target_weights, buckets, lambda x, y: seq2seq_f(x, y, True),
                softmax_loss_function=softmax_loss_function)
            if output_projection is not None:
                for b in xrange(len(buckets)):
                    self.outputs[b] = [
                        tf.matmul(output, output_projection[0]) + output_projection[1]
                        for output in self.outputs[b]
                    ]
        else:
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                self.encoder_inputs, self.decoder_inputs, targets,
                self.target_weights, buckets, lambda x, y: seq2seq_f(x, y, False),
                softmax_loss_function=softmax_loss_function)
        
        # Gradients and SGD update
        params = tf.trainable_variables()
        if not forward_only:
            self.gradient_norms = []
            self.updates = []
            opt = tf.train.GradientDescentOptimizer(self.learning_rate)
            for b in xrange(len(buckets)):
                gradients = tf.gradients(self.losses[b], params)
                clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
                self.gradient_norms.append(norm)
                self.updates.append(opt.apply_gradients(
                    zip(clipped_gradients, params), global_step=self.global_step))
                
        self.saver = tf.train.Saver(tf.global_variables())
    
    def step(self, session, encoder_inputs, decoder_inputs, target_weights, bucket_id, forward_only):
        encoder_size, decoder_size = self.buckets[bucket_id]
        # Input feed
        input_feed = {}
        for l in xrange(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        for l in xrange(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]
        
        # 因為已經將 target weight 作 shift 1, 所以需要補 1 個
        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)
        
        # Output feed
        if not forward_only:
            output_feed = [self.updates[bucket_id], self.gradient_norms[bucket_id], self.losses[bucket_id]]
        else:
            output_feed = [self.losses[bucket_id]]
            for l in xrange(decoder_size):
                output_feed.append(self.outputs[bucket_id][l])
        
        outputs = session.run(output_feed, input_feed)
        if not forward_only:
            return outputs[1], outputs[2], None # Gradient norm, loss, no outputs.
        else:
            return None, outputs[0], outputs[1:] # No gradient norm, loss, outputs.
    
    def get_batch(self, data, bucket_id):
        encoder_size, decoder_size = self.buckets[bucket_id]
        encoder_inputs, decoder_inputs = [], []
        
        # 根據 batch_size 大小 隨機從 data bucket中挑選 encoder 及 decoder 的 inputs
        for _ in xrange(self.batch_size):
            encoder_input, decoder_input = random.choice(data[bucket_id])
            
            # Encoder inputs 放上 "PAD" 符號 以及 reverse
            encoder_pad = [PAD_ID] * (encoder_size - len(encoder_input))
            encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))
            
            # Decoder inputs 前後插上 "GO" 以及 "PAD"
            decoder_pad_size = decoder_size - len(decoder_input) - 1
            decoder_inputs.append([GO_ID] + decoder_input + [PAD_ID] * decoder_pad_size)
        
        # 建立 batch vector from above data
        batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []
        
        # Batch encoder inputs re-indexed encoder_inputs
        for length_idx in xrange(encoder_size):
            batch_encoder_inputs.append(
                np.array([encoder_inputs[batch_idx][length_idx]
                         for batch_idx in xrange(self.batch_size)], dtype=np.int32))
        
        # Batch decoder input re-indexed decoder_inputs, 並建立 weights
        for length_idx in xrange(decoder_size):
            batch_decoder_inputs.append(
                np.array([decoder_inputs[batch_idx][length_idx]
                         for batch_idx in xrange(self.batch_size)], dtype=np.int32))
            
            # 當遇到 "PAD" 符號時，target weight 歸 0
            batch_weight = np.ones(self.batch_size, dtype=np.float32)
            for batch_idx in xrange(self.batch_size):
                # 記住，將 decoder inputs 作 shift 1
                if length_idx < decoder_size-1:
                    target = decoder_inputs[batch_idx][length_idx + 1]
                if length_idx == decoder_size-1 or target == PAD_ID:
                    batch_weight[batch_idx] = 0.0
            batch_weights.append(batch_weight)
        return batch_encoder_inputs, batch_decoder_inputs, batch_weights

def create_model(session, train_dir, usize, vocab_size, num_layers, buckets, max_gradient_norm, batch_size, learning_rate, learning_rate_decay_factor, use_lstm, forward_only):
    dtype = tf.float32
    ckpt = tf.train.get_checkpoint_state(train_dir)
    model = Seq2SeqModel(usize, vocab_size, num_layers, buckets, max_gradient_norm, batch_size, learning_rate, learning_rate_decay_factor, use_lstm, forward_only)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print "Reading model parameters from %s" % ckpt.model_checkpoint_path
    	model.saver.restore(session, ckpt.model_checkpoint_path)
    return model

def initialize_vocabulary(vocabulary_path):
    if gfile.Exists(vocabulary_path):
        rev_vocab = []
        with gfile.GFile(vocabulary_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [tf.compat.as_bytes(line.strip()) for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab

def sentence_to_token_ids(sentence, vocabulary):
    words = jieba.lcut(sentence)
    return [vocabulary.get(w.encode('utf-8'), UNK_ID) for w in words]

def decode(train_dir, usize, vocab_size, num_layers, buckets, max_gradient_norm, batch_size, learning_rate, learning_rate_decay_factor, use_lstm=False, forward_only=True):
    with tf.Session() as session:
        model = create_model(session, 
                             train_dir, 
                             usize,
                             vocab_size,
                             num_layers, 
                             buckets, 
                             max_gradient_norm, 
                             batch_size, 
                             learning_rate, 
                             learning_rate_decay_factor, 
                             use_lstm, 
                             forward_only)
        
        model.batch_size = 1 # decode one sentence at a time
        
        # 載入 vocabulary
        encoder_vocab, decoder_vocab = initialize_vocabulary(vocabulary_path)
        # Decode 使用者輸入的字串
        sys.stdout.write("> ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()
        # TODO: Tokenize 和 mapping to ids
        while sentence:
            # 將 sentence 轉換成 ids 陣列
            token_ids = sentence_to_token_ids(tf.compat.as_bytes(sentence), encoder_vocab)
            bucket_id = len(buckets)-1
            for i, bucket in enumerate(buckets):
                if bucket[0] >= len(token_ids):
                    bucket_id = i
                    break
                else:
                    logging.warning("Sentence truncated: %s", sentence)
            
            # 取 1-element batch to feed sentence to model.
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                {bucket_id: [(token_ids, [])]}, bucket_id)
            
            # 取 output logits for the sentence
            _, _, output_logits = model.step(session, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)
            
            # Greedy decoder - outputs 只是 argmaxes of output_logits
            outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
            
            # output 以 "EOS" 符號為斷點
            if EOS_ID in outputs:
                outputs = outputs[:outputs.index(EOS_ID)]
            # 印出 bot 回覆訊息
            print " ".join([tf.compat.as_str(decoder_vocab[output]) for output in outputs])
            print "> "
            sys.stdout.flush()
            sentence = sys.stdin.readline()

steps_per_checkpoint = 200
train_dir = 'train/'
total_train_steps = 6000
usize=1024
vocab_size=vocabulary_size
num_layers=3
max_gradient_norm=5.0
batch_size=1
learning_rate=0.5
learning_rate_decay_factor=0.99
use_lstm=True
forward_only=True

decode(train_dir, 
       usize,
       vocab_size, 
       num_layers, 
       _buckets, 
       max_gradient_norm, 
       batch_size, 
       learning_rate, 
       learning_rate_decay_factor, 
       use_lstm,
       forward_only)
