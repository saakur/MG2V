# import tensorflow_datasets as tfds
import tensorflow as tf

import time, json, utils, os, sys
import numpy as np
import matplotlib.pyplot as plt
from lib_transformer import *
from config import *
from random import shuffle

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.enable_eager_execution()

class Transformer(tf.keras.Model):
	def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, 
							 target_vocab_size, pe_input, pe_target, rate=0.1):
		super(Transformer, self).__init__()

		self.encoder = Encoder(num_layers, d_model, num_heads, dff, 
													 input_vocab_size, pe_input, rate)

		self.decoder = Decoder(num_layers, d_model, num_heads, dff, 
													 target_vocab_size, pe_target, rate)

		self.final_layer = tf.keras.layers.Dense(target_vocab_size)

	def call(self, inp, tar, training, enc_padding_mask, 
					 look_ahead_mask, dec_padding_mask):

		enc_output = self.encoder(inp, training, enc_padding_mask)	# (batch_size, inp_seq_len, d_model)
		dec_output, attention_weights = self.decoder(
				tar, enc_output, training, look_ahead_mask, dec_padding_mask)
		final_output = self.final_layer(dec_output)	# (batch_size, tar_seq_len, target_vocab_size)

		return final_output, attention_weights, enc_output

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
	def __init__(self, d_model, warmup_steps=40000):
		super(CustomSchedule, self).__init__()

		self.d_model = d_model
		self.d_model = tf.cast(self.d_model, tf.float32)

		self.warmup_steps = warmup_steps

	def __call__(self, step):
		arg1 = tf.math.rsqrt(step)
		arg2 = step * (self.warmup_steps ** -1.5)

		return (tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2))/(1)


def loss_function(real, pred):
	mask = tf.math.logical_not(tf.math.equal(real, 0))
	loss_ = loss_object(real, pred)

	mask = tf.cast(mask, dtype=loss_.dtype)
	loss_ *= mask

	return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


def accuracy_function(real, pred):
	accuracies = tf.equal(real, tf.argmax(pred, axis=2))

	mask = tf.math.logical_not(tf.math.equal(real, 0))
	accuracies = tf.math.logical_and(mask, accuracies)

	accuracies = tf.cast(accuracies, dtype=tf.float32)
	mask = tf.cast(mask, dtype=tf.float32)
	return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)


def create_masks(inp, tar):
	# Encoder padding mask
	enc_padding_mask = create_padding_mask(inp)

	# Used in the 2nd attention block in the decoder.
	# This padding mask is used to mask the encoder outputs.
	dec_padding_mask = create_padding_mask(inp)

	# Used in the 1st attention block in the decoder.
	# It is used to pad and mask future tokens in the input received by 
	# the decoder.
	look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
	dec_target_padding_mask = create_padding_mask(tar)
	combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

	return enc_padding_mask, combined_mask, dec_padding_mask

filePath = "./Data/Pretraining/"
if preProcess:
	trainPath, numFiles = utils.prep_fastq_pretrain(filePath, embedDict, nodeDict)
else:
	trainPath = "./Temp/"

trainFiles_0 = [os.path.join(trainPath, f) for f in os.listdir(trainPath)]
trainFiles_1 = [os.path.join(trainPath, f) for f in os.listdir(trainPath)]

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

transformer = Transformer(num_layers, d_model, num_heads, dff,
                          input_vocab_size, target_vocab_size, 
                          pe_input=MAX_LENGTH,
                          pe_target=MAX_LENGTH,
                          rate=dropout_rate)

learning_rate = CustomSchedule(512)
#learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(1e-5, 10000, 0.96, staircase=False, name=None)
# learning_rate = 1e-4

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9, clipnorm=1.)

ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=25)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
	ckpt.restore(ckpt_manager.latest_checkpoint)
	print ('Latest checkpoint restored!!')

if save_embedding_weights:
	embeddings = transformer.encoder.embedding.get_weights()[0]
	np.save("./Data/cn2v_embedding.npy", embeddings)
# sys.exit(0)

# The @tf.function trace-compiles train_step into a TF graph for faster
# execution. The function specializes to the precise shape of the argument
# tensors. To avoid re-tracing due to the variable sequence lengths or variable
# batch sizes (the last batch is smaller), use input_signature to specify
# more generic shapes.

train_step_signature = [
		tf.TensorSpec(shape=(None, None), dtype=tf.int64),
		tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]

@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
	tar_inp = tar[:, :-1]
	tar_real = tar[:, 1:]

	enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

	with tf.GradientTape() as tape:
		predictions, _, _ = transformer(inp, tar_inp, True, enc_padding_mask, combined_mask, dec_padding_mask)
		loss = loss_function(tar_real, predictions)

	gradients = tape.gradient(loss, transformer.trainable_variables)		
	optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

	train_loss(loss)
	train_accuracy(accuracy_function(tar_real, predictions))


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

steps = 0
mask_perc = 0.15
for epoch in range(EPOCHS):
	start = time.time()

	train_loss.reset_states()
	train_accuracy.reset_states()
	shuffle(trainFiles_1)
	shuffle(trainFiles_0)

	for (i, file_1) in enumerate(trainFiles_1):
		file_0 = trainFiles_0[i%len(trainFiles_0)]
		data_1 = np.load(file_1)
		labels_1 = [1]*data_1.shape[0]
		data_0 = np.load(file_0)
		labels_0 = [1]*data_0.shape[0]
		data = np.reshape(np.vstack([data_1, data_0]), (-1, MAX_LENGTH)) #np.reshape(data_0, (-1, MAX_LENGTH)) # 

		labels = labels_0 + labels_1
		idxs = np.random.permutation(len(labels))
		idxs = np.array_split(idxs, 2500)
		for batch, idx in enumerate(idxs):
			steps += 1
			inp = data[idx][:,:1000]
			tar = data[idx][:,:1000]
			nums = np.ones(inp.shape)
			nums[:, :300] = 0
			np.random.shuffle(np.transpose(nums))
			inp = inp * nums
			tar = tar * (1 - nums)

			train_step(inp, tar)
			if np.isnan(train_loss.result()):
				print(file_0, file_1)
				sys.exit(0)
			if batch % 100 == 0:
				print ('Epoch {} Batch {} Step {} Loss {:.4f} Accuracy {:.4f} LR {}'.format(epoch + 1, i, batch, train_loss.result(), train_accuracy.result(), optimizer._decayed_lr(tf.float32).numpy()))
			if (steps) % 1000 == 0:
				ckpt_save_path = ckpt_manager.save()
				print ('Saving checkpoint for steps {} at {}'.format(steps, ckpt_save_path))
	print ('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, train_loss.result(), train_accuracy.result()))
	print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
