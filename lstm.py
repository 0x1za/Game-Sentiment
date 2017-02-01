# Import dependencies
import tflearn
import pandas as pd
from tflearn.data_utils import pad_sequences, to_categorical
from sklearn.model_selection import train_test_split

#Load the gaming dataset
df = pd.read_pickle('ign.pkl')

#Convert the editor choice values to boolean values
df['editors_choice'] = df['editors_choice'].replace('N', int(0))
df['editors_choice'] = df['editors_choice'].replace('Y', int(1))

#Convert dtypes to required
df['score'] = df['score'].astype(int)
df['platform'] = df['platform'].astype(str)
df['score_phrase'] = df['score_phrase'].astype(str)
df['release_year'] = df['release_year'].astype(int)

#Split into X and y
X = df[['platform', 'score', 'genre', 'score_phrase']]
y = df[['editors_choice']]

#Split X and y intp train and test sets
trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.33, random_state=42)

#Data preprocessing
#Sequence padding
trainX = pad_sequences(trainX, maxlen=None, dtype='str', padding='post', truncating='post', value=0.)
testX = pad_sequences(testX, dtype='str', maxlen=100, padding='post', truncating='post', value=0.)
# Converting labels to binary vectors
trainY = to_categorical(trainY, nb_classes=2)
testY = to_categorical(testY, nb_classes=2)



# Network building
net = tflearn.input_data([None, 100])
net = tflearn.embedding(net, input_dim=10000, output_dim=128)
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy')

# Training
col = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
for x in col:
    tf.add_to_collection(tf.GraphKeys.VARIABLES, x)

model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True,
          batch_size=32)
