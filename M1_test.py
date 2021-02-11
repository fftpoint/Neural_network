# Import Packages
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# 学習回数
epochs=100
# 中間層ユニット数
num_hidden = 3

# データ生成
x = np.arange(-10, 10, 0.1) 
# input_data = (x - x.min()) / (x.max() - x.min())
input_data = x  / x.max()
# input_data = (x - x.mean()) / x.std()
training_data = 4 * input_data ** 3 - 2 * input_data
# training_data = (y - y.min()) / (y.max() - y.min())

"""
# 教師データグラフ出力
plt.title('train_data')
plt.plot(input_data, training_data, linestyle="dashed")
plt.savefig('images/train_data.png')
plt.show()
"""

# モデル生成
model = keras.Sequential(
    [   
        layers.InputLayer(input_shape=(1,)),
        layers.Dense(num_hidden, activation="sigmoid"),
        layers.Dense(1, activation="linear")
    ]
)
model.summary()

# 学習方法
model.compile(
    loss='mse', # 誤差関数、平均二乗誤差
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.1), # 確率的勾配降下法
    metrics='mse' # 評価関数、平均二乗誤差
)

# 学習
print("Fit model on training data")
history = model.fit(
    input_data,
    training_data,
    batch_size=1,
    epochs=epochs,
    # validation_data = (input_data,training_data)
)

# 検証
print("Evaluate on test data")
results = model.predict(input_data)
# print("test loss, test acc:", results)
