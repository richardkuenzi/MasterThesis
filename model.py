import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import numpy as np
import matplotlib.pyplot as plt

# change to your needs
fold = '1'
label = '00.0010'
epochs = 15

# Train Dataset
medical = pd.read_csv("C:/Users/richa/Dropbox/FHNW/MSc/Thesis/Dataset/" + label + "/train_fold_" + fold + ".csv")
medical = medical.sample(frac=1).reset_index(drop=True)

# View at CSV to make sure it's alright
print(medical.head(3))

preprocessed_train_inputs = []
preprocessed_test_inputs = []

medical_features = medical.copy()
medical_labels = medical_features.pop(label)

# Test Dataset
proof = pd.read_csv("C:/Users/richa/Dropbox/FHNW/MSc/Thesis/Dataset/" + label + "/val_fold_" + fold + ".csv")
proof.head()
proof = proof.sample(frac=1).reset_index(drop=True)


proof_features = proof.copy()
proof_labels = proof_features.pop(label)


inputs = {}

# Create Input Tensor for Train & Test
for name, column in medical_features.items():
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


for column in medical_features:
    if medical_features[column].dtype == np.float64 or medical_features[column].dtype == np.int64:
        preprocessed_train_inputs.insert(0, medical_features[column])
    else:
        preprocessed_train_inputs.append(medical_features[column])

numeric_inputs = {name:input for name,input in inputs.items()
                  if input.dtype==tf.float32}

# Concatenate and Normalization of Numerical input
x = layers.Concatenate()(list(numeric_inputs.values()))
norm = preprocessing.Normalization()
norm.adapt(np.array(medical[numeric_inputs.keys()]))
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

    # Create dataset and layers for text input
    data = input
    layer = preprocessing.TextVectorization(max_tokens=max_features,output_mode="int",output_sequence_length=sequence_length)
    dataset = tf.data.Dataset.from_tensor_slices((medical_features['txt'], medical_labels)).batch(32)
    text_ds = dataset.map(lambda x, y: x)
    layer.adapt(text_ds)
    vectorized_text = layer(data)
    
    x = layers.Embedding(max_features + 1, embedding_dim)(vectorized_text)
    x = layers.Dropout(0.5)(x)

    # Change to switch between CNN, LSTM or GRU
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

# Concatenate numeric and text inputs
all_predictions = layers.Concatenate()([all_numeric_inputs, all_text_inputs])
all_predictions = layers.Dense(1, activation="sigmoid")(all_predictions)

medical_features_dict = {name: np.array(value)
                         for name, value in medical_features.items()}

medical_model = tf.keras.Model(inputs, all_predictions)
medical_model.compile(loss="binary_crossentropy", optimizer="Adam", metrics="accuracy")
history = medical_model.fit(x=preprocessed_train_inputs, y=medical_labels, epochs=epochs)
medical_model.evaluate(x=preprocessed_test_inputs, y=proof_labels)

tf.keras.utils.plot_model(model = medical_model , rankdir="LR", dpi=72, show_shapes=True)



# Writing out_fold_X into list for further analysis
csv_list = [[], [], []]

for element in preprocessed_test_inputs[115]:
    append_category = csv_list[0]
    append_category.append(element)

for element in proof_labels:
    append_category = csv_list[1]
    append_category.append(element)

predictions = medical_model.predict(preprocessed_test_inputs).reshape(-1,)

for prediction in predictions:
    append_prediction = csv_list[2]
    append_prediction.append(prediction)

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
