# 24 - 3 -2 -1
# 通过卷积和池化层，由24维度变为1维，chanenls变为128
# 第一层：3个一维卷积层：4*32，3*64，2*128
# 第二层：最大池化层 ：stride 4,3,2
# 第三层：全连接层 1000
# 第四层：全连接层 500
# 第五层：softmax层