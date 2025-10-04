# src/model_fno.py
"""Fourier Neural Operator (1D over time) for multi-step water level forecasting.
The module implements a lightweight 1D FNO variant that performs spectral
convolution along the time axis and outputs a 7-step sequence for water level (h).
"""
import tensorflow as tf

class SeasonalFNO1D(tf.keras.Model):
    """1D Fourier Neural Operator over time with dropout/L2 regularization.

    The model:
      inputs  -> Dense(width) -> [FourierLayer + Dense(width) + GELU]*L -> Dense(128) -> Dense(7)
      outputs -> last-time-step 7-step sequence for h only: shape [B, 7, 1]

    Args:
        modes: Number of retained Fourier modes along time (upper bound).
        width: Hidden channel width after the input Dense.
        num_layers: Number of Fourier+pointwise blocks.
        input_features: Number of input feature channels per time step (informational).
        dropout_rate: Dropout rate applied after key blocks.
        l2: L2 regularization factor for Dense layers.

    Notes:
        - The spectral mixing uses rFFT on the time axis. We truncate to at most
          `modes` frequencies and apply a learned complex linear map per frequency.
        - The final Dense predicts 7 values per time step; we reshape to [B, T, 7, 1]
          and return the last time step => [B, 7, 1].
    """

    def __init__(self, modes=32, width=64, num_layers=4, input_features=6,
                 dropout_rate=0.1, l2=1e-5):
        super().__init__()
        self.modes = modes
        self.width = width
        self.num_layers = num_layers
        self.input_features = input_features
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        self.fc0 = tf.keras.layers.Dense(
            width, kernel_regularizer=tf.keras.regularizers.l2(l2))

        self.fourier_weights, self.conv_layers = [], []
        for i in range(num_layers):
            wr = self.add_weight(shape=(modes, width, width),
                                 initializer='glorot_normal',
                                 name=f'fourier_weight_real_{i}')
            wi = self.add_weight(shape=(modes, width, width),
                                 initializer='glorot_normal',
                                 name=f'fourier_weight_imag_{i}')
            self.fourier_weights.append((wr, wi))
            self.conv_layers.append(
                tf.keras.layers.Dense(
                    width, kernel_regularizer=tf.keras.regularizers.l2(l2)
                )
            )
        self.fc1 = tf.keras.layers.Dense(
            128, kernel_regularizer=tf.keras.regularizers.l2(l2)
        )

        # ===== Output 7-step sequence for h only =====
        self.fc2 = tf.keras.layers.Dense(
            7 * 1, kernel_regularizer=tf.keras.regularizers.l2(l2) # 7 days × 1 var (h)
        )

    def fourier_layer_time(self, x, weights_real, weights_imag):
        """Spectral mixing along time using learned complex weights.

        Args:
            x: Tensor of shape [B, T, C] (float32).
            weights_real: Tensor of shape [modes, C, C] (float32).
            weights_imag: Tensor of shape [modes, C, C] (float32).

        Returns:
            Tensor of shape [B, T, C], same dtype as x (float32).
        """
        # x: [B, T, C]
        x_tc = tf.transpose(x, [0, 2, 1])   # [B, C, T]
        x_ft = tf.signal.rfft(tf.cast(x_tc, tf.float32))    # [B, C, T//2+1]
        B = tf.shape(x_ft)[0]; C = tf.shape(x_ft)[1]; Tfreq = tf.shape(x_ft)[2]
        M = tf.minimum(self.modes, Tfreq)
        x_ft_trunc = x_ft[:, :, :M] # [B, C, M]

        w = tf.complex(weights_real[:M, :, :], weights_imag[:M, :, :])  # [M, C, C]
        out_ft = tf.einsum('bcm,mco->bom', x_ft_trunc, w)   # [B, C, M]

        pad_len = Tfreq - M
        out_ft_padded = tf.concat([out_ft, tf.zeros([B, C, pad_len], dtype=out_ft.dtype)], axis=2)
        T = tf.shape(x)[1]
        x_time = tf.signal.irfft(out_ft_padded, fft_length=[T]) # [B, C, T]
        return tf.transpose(x_time, [0, 2, 1])  # [B, T, C]

    def call(self, inputs, training=False):
        """Forward pass.

        Args:
            inputs: Tensor of shape [B, T, F] (float32), F = input_features.
            training: Bool flag for dropout.

        Returns:
            Tensor of shape [B, 7, 1] — the 7-step sequence from the last time step.
        """

        x = self.fc0(inputs); x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x_fourier = self.fourier_layer_time(x, self.fourier_weights[i][0], self.fourier_weights[i][1])
            x = tf.nn.gelu(x_fourier + self.conv_layers[i](x))
            x = self.dropout(x, training=training)
        x = tf.nn.gelu(self.fc1(x))
        x = self.dropout(x, training=training)

        # ===== Output 7-day sequence for h only =====
        out = self.fc2(x)                                # [B, T, 7]
        out = tf.reshape(out, [-1, tf.shape(x)[1], 7, 1])   # [B, T, 7, 1]
        return out[:, -1, :, :]                          # take the last time step -> [B, 7, 1]
