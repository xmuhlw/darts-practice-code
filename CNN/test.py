import numpy as np
train_data = [1,2,3,4,5,6,3,4]
num_train = len(train_data)
indices = list(range(num_train))
split = int(np.floor(0.5 * num_train))
print(split)