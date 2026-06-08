from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam


def build_mlp(input_dim):
    """
    Build and compile a feed-forward neural network (MLP) for binary classification.

    The architecture is optimized for tabular data and includes multiple dense
    layers with ReLU activation, dropout regularization, and a sigmoid output
    for binary classification.

    Parameters
    ----------
    input_dim : int
        Number of input features.

    Returns
    -------
    tensorflow.keras.Model
        A compiled Keras model ready for training.
    """
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0002),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model
