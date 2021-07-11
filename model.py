import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import numpy as np
import matplotlib.pyplot as plt

fold = '1'
label = '00.0010'
epochs = 2

# Train Dataset
titanic = pd.read_csv("C:/Users/richa/Dropbox/FHNW/MSc/Thesis/Dataset/" + label + "/train_fold_" + fold + ".csv")
titanic = titanic.sample(frac=1).reset_index(drop=True)

print(titanic.head(3))

preprocessed_text_inputs = []
preprocessed_test_inputs = []

titanic_features = titanic.copy()
titanic_labels = titanic_features.pop(label)

# Test Dataset
proof = pd.read_csv("C:/Users/richa/Dropbox/FHNW/MSc/Thesis/Dataset/" + label + "/val_fold_" + fold + ".csv")
proof.head()
proof = proof.sample(frac=1).reset_index(drop=True)


proof_features = proof.copy()
proof_labels = proof_features.pop(label)


inputs = {}

for name, column in titanic_features.items():
  dtype = column.dtype
  if dtype == object:
    dtype = tf.string
  else:
    dtype = tf.float32

  inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)
  
proofs = {}

for name, column in proof_features.items():
  dtype = column.dtype
  if dtype == object:
    dtype = tf.string
  else:
    dtype = tf.float32

  proofs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)


for column in proof_features:
    if proof_features[column].dtype == np.float64 or proof_features[column].dtype == np.int64:
        preprocessed_test_inputs.insert(0, proof_features[column])
    else:
        preprocessed_test_inputs.append(proof_features[column])


for column in titanic_features:
    if titanic_features[column].dtype == np.float64 or titanic_features[column].dtype == np.int64:
        preprocessed_text_inputs.insert(0, titanic_features[column])
    else:
        preprocessed_text_inputs.append(titanic_features[column])

numeric_inputs = {name:input for name,input in inputs.items()
                  if input.dtype==tf.float32}


x = layers.Concatenate()(list(numeric_inputs.values()))
norm = preprocessing.Normalization()
norm.adapt(np.array(titanic[numeric_inputs.keys()]))
all_numeric_inputs = norm(x)

# Model constants.
max_features = 20000
embedding_dim = 128
sequence_length = 500

# Now that we have our custom standardization, we can instantiate our text
# vectorization layer. We are using this layer to normalize, split, and map
# strings to integers, so we set our 'output_mode' to 'int'.
# Note that we're using the default split function,
# and the custom standardization defined above.
# We also set an explicit maximum sequence length, since the CNNs later in our
# model won't support ragged sequences.
vectorize_layer = TextVectorization(
    max_tokens=max_features,
    output_mode="int",
    output_sequence_length=sequence_length,
)


for name, input in inputs.items():
  if input.dtype == tf.float32:
    continue
  if input.dtype == tf.string:
    
    data = input
    layer = preprocessing.TextVectorization(max_tokens=max_features,output_mode="int",output_sequence_length=sequence_length)
    dataset = tf.data.Dataset.from_tensor_slices((titanic_features['txt'], titanic_labels)).batch(32)
    text_ds = dataset.map(lambda x, y: x)
    layer.adapt(text_ds)
    vectorized_text = layer(data)
    
    x = layers.Embedding(max_features + 1, embedding_dim)(vectorized_text)
    x = layers.Dropout(0.5)(x)
    # CNN
    x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
    x = layers.GlobalMaxPooling1D()(x)

    # LSTM
    #x = layers.Bidirectional(layers.LSTM(128))(x)

    # GRU
    #x = layers.Bidirectional(layers.GRU(128, return_sequences=True))(x)
    #x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)
    all_text_inputs = layers.Dropout(0.5)(x)


all_predictions = layers.Concatenate()([all_numeric_inputs, all_text_inputs])
all_predictions = layers.Dense(1, activation="sigmoid")(all_predictions)

titanic_features_dict = {name: np.array(value) 
                         for name, value in titanic_features.items()}

titanic_preprocessing = tf.keras.Model(inputs, all_predictions)

titanic_preprocessing.compile(loss="binary_crossentropy", optimizer="Adam", metrics="accuracy")


history = titanic_preprocessing.fit(x=preprocessed_text_inputs, y=titanic_labels, epochs=epochs)

titanic_preprocessing.evaluate(x=preprocessed_test_inputs, y=proof_labels)


tf.keras.utils.plot_model(model = titanic_preprocessing , rankdir="LR", dpi=72, show_shapes=True)



# Writing test_ds into list for further analysis
csv_list = [[], [], []]

for element in preprocessed_test_inputs[115]:
    append_category = csv_list[0]
    append_category.append(element)

for element in proof_labels:
    append_category = csv_list[1]
    append_category.append(element)

predictions = titanic_preprocessing.predict(preprocessed_test_inputs).reshape(-1,)

for prediction in predictions:
    append_prediction = csv_list[2]
    append_prediction.append(prediction)

#print(csv_list)
import pandas as pd
df = pd.DataFrame(csv_list)
df.to_csv("C:/Users/richa/Dropbox/FHNW/MSc/Thesis/Dataset/" + label + "/out_fold_" + fold + ".csv", index=False, header=False, sep="§")


# Plot figures
plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()
