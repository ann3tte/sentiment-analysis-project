# GOOGLE COLAB WAS USED, COPY AND PASTE THE FOLLWOING INTO A CELL


# To install required libraries, uncomment the line below:
# !pip install tensorflow nltk scikit-learn --quiet

import numpy as np
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from utils import LABEL_MAP, load_dataset, split_data


# --- Enhanced GloVe Loader ---
def load_glove(filename):
    words = {}
    with open(filename, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            word = parts[0]
            vector = np.array(parts[1:], dtype="float32")
            words[word] = vector
    return words

# Modify path if using different trained word embeddings:
words = load_glove("/content/drive/MyDrive/ProjectDatasets/glove.6B.50d.txt")

# --- LSTM Model Builder ---
def build_lstm_model(vocab_size, embedding_dim, max_length, embedding_matrix):
    model = Sequential([
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=max_length,
            weights=[embedding_matrix],
            trainable=True,
        ),
        Bidirectional(LSTM(
            64,
            dropout=0.3,
            recurrent_dropout=0.3,
            return_sequences=False,
        )),
        Dense(32, activation="relu"),
        Dense(3, activation="softmax"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0005),   # LEARNING RATE
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

# --- Main Training Loop ---
# Modify the path if using different datasets:
datasets = [
    ("/content/drive/MyDrive/ProjectDatasets/tweets.csv", "Tweets"),
    ("/content/drive/MyDrive/ProjectDatasets/youtube.csv", "YouTube"),
    ("/content/drive/MyDrive/ProjectDatasets/amazon.csv", "Amazon"),
]

for dataset_path, dataset_name in datasets:
    print(f"\n{'='*50}")
    print(f"Processing {dataset_name} Dataset")
    print(f"{'='*50}")

    # Data Loading
    data = load_dataset(dataset_path)
    train_data, val_data, test_data = split_data(data)

    # Prepare features and labels
    X_train = [text for text, _ in train_data]
    y_train = np.array([label for _, label in train_data])

    # Class Weight Calculation
    class_weights = compute_class_weight(
        "balanced",
        classes=np.unique(y_train),
        y=y_train,
    )
    class_weights = {i:w for i,w in enumerate(class_weights)}
    print("Class weights:", class_weights)

    # Tokenization
    tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")  # noqa: S106
    tokenizer.fit_on_texts(X_train)

    max_length = 100
    X_train = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=max_length)
    X_val = pad_sequences(tokenizer.texts_to_sequences(
        [text for text, _ in val_data]), maxlen=max_length)
    y_val = np.array([label for _, label in val_data])

    # Embeddings
    embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, 50))
    for word, i in tokenizer.word_index.items():
        if word in words:
            embedding_matrix[i] = words[word]
        else:
            embedding_matrix[i] = np.random.normal(size=(50,))

    # Model Training
    model = build_lstm_model(
        vocab_size=len(tokenizer.word_index) + 1,
        embedding_dim=50,
        max_length=max_length,
        embedding_matrix=embedding_matrix,
    )

    history = model.fit(
        X_train, y_train,
        epochs=10,          # EPOCHS
        batch_size=128,     # BATCH SIZE
        validation_data=(X_val, y_val),
        class_weight=class_weights,
        callbacks=[
            EarlyStopping(patience=3, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=2),
        ],
        verbose=2,
    )

    # Evaluation
    X_test = pad_sequences(tokenizer.texts_to_sequences(
        [text for text, _ in test_data]), maxlen=max_length)
    y_test = np.array([label for _, label in test_data])

    test_predictions = np.argmax(model.predict(X_test), axis=1)

    print(f"\n{dataset_name} Classification Report:")
    print(classification_report(
        y_test, test_predictions,
        target_names=LABEL_MAP.keys(),
    ))
