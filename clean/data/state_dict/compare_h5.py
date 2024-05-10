import h5py
import os


def h5_files_are_equal(file1, file2):
    if not os.path.isfile(file1) or not os.path.isfile(file2):
        return False

    with h5py.File(file1, 'r') as h51, h5py.File(file2, 'r') as h52:
        if set(h51.keys()) != set(h52.keys()):
            return False

        for key in h51.keys():
            if not h5_datasets_are_equal(h51[key], h52[key]):
                return False

    return True


def h5_datasets_are_equal(dataset1, dataset2):
    if dataset1.name != dataset2.name:
        return False
    # if dataset1.shape != dataset2.shape:
    #     return False
    # if dataset1.dtype != dataset2.dtype:
    #     return False
    # if (dataset1[:] != dataset2[:]).any():
        # return False

    return True


# 使用方法：
# 将文件路径替换为您要比较的H5文件
file1 = r'C:\Users\ASUS\Downloads\inceptiontime new 150 15 (2)\model.weights.h5'
file2 = r'C:\Users\ASUS\Downloads\inceptiontime new 150 15 (1)\model.weights.h5'

print(h5_files_are_equal(file1, file2))


# 导入 hashlib 模块，以便使用哈希函数
import hashlib

# 定义一个函数，用于计算文件的哈希值
def calculate_file_hash(file_path, hash_algorithm="sha256"):
    # 创建一个哈希对象，使用指定的哈希算法
    hash_obj = hashlib.new(hash_algorithm)

    # 打开文件以二进制只读模式
    with open(file_path, "rb") as file:
        while True:
            # 从文件中读取数据块（64 KB大小）
            data = file.read(65536)  # 64 KB buffer
            if not data:
                break

            # 更新哈希对象，将数据块添加到哈希值中
            hash_obj.update(data)

    # 返回哈希值的十六进制表示
    return hash_obj.hexdigest()

# 指定要比较的两个文件的路径
file1 = r'C:\Users\ASUS\Downloads\inceptiontime new 150 15 (2)\model.weights.h5'
file2 = r'C:\Users\ASUS\Downloads\inceptiontime new 150 15 (1)\model.weights.h5'

# 使用哈希函数计算文件1的哈希值
hash1 = calculate_file_hash(file1)

# 使用哈希函数计算文件2的哈希值
hash2 = calculate_file_hash(file2)

# 比较两个哈希值，如果相同，表示文件内容相同
if hash1 == hash2:
    print("两个文件相同")
else:
    print("两个文件不同")
