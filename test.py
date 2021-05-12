# coding=utf-8

from PIL import Image
import numpy as np
import os
from keras.layers import Dense
from keras.preprocessing.image import load_img
from keras.preprocessing import image
from tqdm import tqdm
from keras.models import load_model
import time

# 测试图像路径
detect_path = "./test/dujiahao1"

# 标签
name_lst = ['chenbaohua1', 'chenbaohua2', 'chengang', 'chenggang2', 'chenlei1', 'chenlei2', 'chenzhaohua2',
            'chenzhaoihua1',
            'chenzhiqiang1', 'chenzhiqiang2', 'dengpengwei1', 'dengpengwei2', 'dujiahao1', 'dujiahao2',
            'huangxiaohang1',
            'huangxiaohang2', 'lanqiaoke1', 'lanqiaoke2', 'liuxiang1', 'liuxiang2', 'qinqi1', 'qinqi2', 'qinsong1',
            'qinsong2', 'quzihan1', 'quzihan2', 'shechanchen1', 'shechanchen2', 'tanyuanyong1', 'tanyuanyong2', 'wudi1',
            'wudi2', 'yepeng1', 'yepeng2', 'yumingwei1', 'yumingwei2', 'zhangboyu1', 'zhangboyu2', 'zhouxudong1',
            'zhouxudong2']


class Test():
    def load_model(self, model_type):
        model = None
        if model_type == 'vgg16':
            model = load_model('model/vgg16_weights.h5')
        elif model_type == 'resnet50':
            model = load_model('model/restnet50_weights.h5')
        return model

    def main(self, model_type):
        model = self.load_model(model_type=model_type)
        start_time = time.time()
        with open(os.path.join('./result', '{}_result.txt').format(model_type), 'w') as f:
            for root, dirs, filenames in os.walk(detect_path):
                for filename in tqdm(sorted(filenames, key=lambda x: int(x.split('.')[0][2])), desc='开始预测测试集'):
                    filepath = os.path.join(detect_path, filename)
                    img = load_img(filepath, target_size=(224, 224))
                    img = image.img_to_array(img) / 255.0
                    print(img.shape)
                    img = np.expand_dims(img, axis=0)
                    print(img.shape)
                    print()
                    predictions = model.predict(img)
                    predict_label = np.argmax(predictions[0])
                    # if predictions[0][0] > predictions[0][1]:
                    #     result = 'NO'
                    # else:
                    #     result = 'YES'
                    f.write('{} {}\n'.format(filename, name_lst[predict_label]))
        end_time = time.time()
        print("耗时：", end_time - start_time)


if __name__ == '__main__':
    print("网络结构：vgg16,resnet50")
    model_type = str(input("选择输入网络结构:"))
    test = Test()
    test.main(model_type=model_type)
