import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
from keras.utils import plot_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from random import shuffle

batch_size = 64  # Batch size for training.
epochs = 500  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 500  # Number of samples to train on.
# Path to the data txt file on disk.
data_path = 'spa-eng/spa.txt'
num_words = 2000

# Vectorize the data.
input_texts = []
target_texts = []
lines = open(data_path).read().split('\n')
shuffle(lines)
for line in lines[: min(num_samples, len(lines) - 1)]:
    input_text, target_text = line.split('\t')
    target_text = '§START ' + target_text + ' §STOP'
    input_texts.append(input_text)
    target_texts.append(target_text)

input_tokenizer = Tokenizer(num_words=num_words, lower=False, split=' ', filters='')
input_tokenizer.fit_on_texts(input_texts)
input_word_index = input_tokenizer.word_index
target_tokenizer = Tokenizer(num_words=num_words, lower=False, split=' ', filters='')
target_tokenizer.fit_on_texts(target_texts)
target_word_index = target_tokenizer.word_index

num_encoder_tokens = num_words # len(input_word_index)
num_decoder_tokens = num_words # len(target_word_index)
max_encoder_seq_length = max([len(txt.split(' ')) for txt in input_texts])
max_decoder_seq_length = max([len(txt.split(' ')) for txt in target_texts])

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

encoder_input_data = pad_sequences(input_tokenizer.texts_to_sequences(input_texts), maxlen=max_encoder_seq_length)
decoder_input_data = pad_sequences(target_tokenizer.texts_to_sequences(target_texts), maxlen=max_decoder_seq_length)
decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')

for i, target_text in enumerate(target_texts):
    for t, word in enumerate(target_text.split(' ')):
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_word_index[word] - 1] = 1.

# Define an input sequence and process it.
encoder_inputs = Input(shape=(None,))
x = Embedding(num_encoder_tokens, latent_dim)(encoder_inputs)
x, state_h, state_c = LSTM(latent_dim, return_state=True)(x)
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None,))
x_emb = Embedding(num_decoder_tokens, latent_dim)(decoder_inputs)
x_tmp = LSTM(latent_dim, return_state=True, return_sequences=True)
x, _, _ = x_tmp(x_emb, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(x)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compile & run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
# Note that `decoder_target_data` needs to be one-hot encoded,
# rather than sequences of integers like `decoder_input_data`!
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size, epochs=epochs)

# Save model
model.save('translator.h5')

# Next: inference mode (sampling).
# Here's the drill:
# 1) encode input and retrieve initial decoder state
# 2) run one step of decoder with this initial state
# and a "start of sequence" token as target.
# Output will be the next target token
# 3) Repeat with the current target token and current states

# Define sampling models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = x_tmp(x_emb, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_word_index = dict(
    (i, word) for word, i in input_word_index.items())
reverse_target_word_index = dict(
    (i, word) for word, i in target_word_index.items())


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.array([[target_word_index['§START']]]) #np.zeros((1,num_decoder_tokens))
    # # Populate the first character of target sequence with the start character.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    sentence_length = 0
    while not stop_condition:
        output_words, h, c = decoder_model.predict([target_seq] + states_value)

        # Sample a token
        sampled_word_index = np.argmax(output_words[0, -1, :])
        sampled_word = reverse_target_word_index[sampled_word_index+1] # TODO check
        decoded_sentence += ' ' + sampled_word
        sentence_length += 1
        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_word == '§STOP' or
           sentence_length > max_decoder_seq_length):
            stop_condition = True

        target_seq = np.array([[target_word_index[sampled_word]]]) 

        # Update states
        states_value = [h, c]

    return decoded_sentence[:-5]

usr = input()

while usr != 'exit' and usr != 'quit':
    if usr == 'test':
        for seq_index in range(20):
            # Take one sequence (part of the training test) for trying out decoding.
            input_seq = encoder_input_data[seq_index: seq_index + 1]
            decoded_sentence = decode_sequence(input_seq)
            print()
            print('Input sentence:', input_texts[seq_index])
            print('Decoded sentence:', decoded_sentence)
    else:
        input_texts.append(usr)
        encoder_input_data = pad_sequences(input_tokenizer.texts_to_sequences(input_texts), maxlen=max_encoder_seq_length)
        input_seq = encoder_input_data[len(input_texts)-1 : len(input_texts)]
        breakpoint()
        decoded_sentence = decode_sequence(input_seq)
        print('Input sentence:', usr)
        print('Decoded sentence:', decoded_sentence)
    usr = input()

print('exiting...')
