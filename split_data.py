from sklearn.model_selection import train_test_split
from sklearn import datasets
import os
import shutil

# 分类列表
name_list = ["shechanchenyou", "shechanchenzuo", "yepengyou", "yepengzuo"]

# 未划分的数据集文件夹
base_dir=r'/or_data/'
# 划分的数据集保存文件夹
base_save_dir = r"/data/"
for f_name in name_list:
    print(f_name)
    dir_path = base_dir+f_name+"/"
    dirs = os.listdir(dir_path)
    print(dirs)
    train_set, test_set = train_test_split(dirs, test_size=0.2, random_state=42)
    for d in test_set:
        print(dir_path+d)
        print(base_save_dir+"test/"+f_name+"/")
        shutil.copy(dir_path+d, base_save_dir+"test/"+f_name+"/")
        shutil.copy(dir_path+d, base_save_dir+"val/"+f_name+"/")
    for d in train_set:
        shutil.copy(dir_path+d, base_save_dir+"train/"+f_name+"/")

