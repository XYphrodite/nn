import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Обучающее множество
X_train = np.array([[0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 1, 1],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 1],
                    [0, 0, 1, 1, 0],
                    [0, 0, 1, 1, 1],
                    [0, 1, 0, 0, 0],
                    [0, 1, 0, 0, 1],
                    [0, 1, 0, 1, 0],
                    [0, 1, 0, 1, 1],
                    [0, 1, 1, 0, 0],
                    [0, 1, 1, 0, 1],
                    [0, 1, 1, 1, 0],
                    [0, 1, 1, 1, 1]
                    ])
y_train = np.array([[0], [1], [0], [1], [0], [1], [0], [1], [0], [1], [0], [1], [0], [1], [0], [1]])

# Определение архитектуры нейронной сети
model = tf.keras.Sequential([
    tf.keras.layers.Dense(2, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(3, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Вывод начальных весов
for i, layer in enumerate(model.layers):
    print(f"Layer {i + 1}:")
    print(f"Initial Weights:\n{layer.get_weights()[0]}")
    print(f"Initial Biases:\n{layer.get_weights()[1]}")
    print()

# Компиляция модели
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Список для хранения изменения весов
weight_history = []

# Обратный вызов для сохранения весов только каждую сотую эпоху
class SaveWeights(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 100 == 0:
            weights = []
            for layer in model.layers:
                weights.append(layer.get_weights()[0])
            weight_history.append(weights)

# Обучение модели с использованием обратного вызова для сохранения весов
model.fit(X_train, y_train, epochs=1000, verbose=0, callbacks=[SaveWeights()])

# Вывод конечных весов
for i, layer in enumerate(model.layers):
    print(f"Layer {i + 1}:")
    print(f"Final Weights:\n{layer.get_weights()[0]}")
    print(f"Final Biases:\n{layer.get_weights()[1]}")
    print()

# Пример использования модели
X_new = np.array([[0,0, 0, 1, 1]])
predictions = model.predict(X_new)
print(predictions)

# Визуализация изменения весов
for layer_num, weights in enumerate(zip(*weight_history)):
    plt.figure(figsize=(8, 6))
    for i, layer_weights in enumerate(weights):
        plt.plot(layer_weights.flatten(), label=f'Epoch {i + 1}')
    plt.title(f'Layer {layer_num + 1} Weights')
    plt.xlabel('Weight Index')
    plt.ylabel('Weight Value')
    plt.legend()
    plt.grid(True)
    plt.show()



print(weight_history)
print(weights)