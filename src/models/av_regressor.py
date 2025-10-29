# Arousal–Valence LSTM Regressor (TensorFlow/Keras)
# Outputs continuous arousal and valence in [0, 1].

from typing import Optional, Tuple
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
import numpy as np


class AVLSTMRegressor:
    """
    Bidirectional LSTM regressor predicting continuous arousal and valence.
    Input shape: (timesteps, num_features)
    Output: 2 floats in [0, 1] => [arousal, valence]
    """

    def __init__(
        self,
        input_shape: Tuple[int, int],
        lstm_units: int = 96,
        dropout_rate: float = 0.3,
        learning_rate: float = 1e-3,
    ) -> None:
        self.input_shape = input_shape
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model: Optional[tf.keras.Model] = None
        self.history = None

    def build(self) -> tf.keras.Model:
        inputs = layers.Input(shape=self.input_shape)

        x = layers.Bidirectional(layers.LSTM(self.lstm_units, return_sequences=True))(inputs)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.BatchNormalization()(x)

        x = layers.Bidirectional(layers.LSTM(self.lstm_units // 2))(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.BatchNormalization()(x)

        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(64, activation="relu")(x)

        # Output bounded to [0, 1]
        outputs = layers.Dense(2, activation="sigmoid", name="av_output")(x)

        model = models.Model(inputs=inputs, outputs=outputs, name="AV_LSTM_Regressor")
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss="mse",
            metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
        )

        self.model = model
        return model

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        verbose: int = 1,
    ):
        if self.model is None:
            self.build()

        cbs = [
            callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1),
        ]

        self.history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=cbs,
            verbose=verbose,
        )
        return self.history

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray):
        if self.model is None:
            raise ValueError("Model not built/trained.")
        return self.model.evaluate(X_test, y_test, verbose=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not built/trained.")
        return self.model.predict(X)

    def save(self, path: str) -> None:
        if self.model is None:
            raise ValueError("Model not built/trained.")
        self.model.save(path)

    def load(self, path: str) -> None:
        self.model = tf.keras.models.load_model(path)
