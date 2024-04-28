import numpy as np
from sklearn.preprocessing import OneHotEncoder

y = np.array([1,7,8])
print(y)
y = y.reshape(-1,1)
print(y)
onehot_encoder = OneHotEncoder(categories=[[0,1,2,3,4,5,6,7,8]], sparse_output=False)
new_y = onehot_encoder.fit_transform(y)
print(new_y)