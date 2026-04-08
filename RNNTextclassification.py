#RNN LSTM
#Text Classification using Simple RNN
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
vocab_size = 10000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

max_len = 200
X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

model_rnn = Sequential()
model_rnn.add(Embedding(input_dim=vocab_size, output_dim=128, input_length=max_len))
model_rnn.add(SimpleRNN(64))
model_rnn.add(Dense(1, activation='sigmoid'))

model_rnn.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

model_rnn.fit(X_train, y_train, epochs=3, batch_size=64, validation_split=0.2)


loss, acc = model_rnn.evaluate(X_test, y_test)
print("RNN Test Accuracy:", acc)
