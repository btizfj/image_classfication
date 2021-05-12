import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import PIL
import copy

# 标签
name_lst = ['chenbaohua1', 'chenbaohua2', 'chengang', 'chenggang2', 'chenlei1', 'chenlei2', 'chenzhaohua2',
            'chenzhaoihua1',
            'chenzhiqiang1', 'chenzhiqiang2', 'dengpengwei1', 'dengpengwei2', 'dujiahao1', 'dujiahao2',
            'huangxiaohang1',
            'huangxiaohang2', 'lanqiaoke1', 'lanqiaoke2', 'liuxiang1', 'liuxiang2', 'qinqi1', 'qinqi2', 'qinsong1',
            'qinsong2', 'quzihan1', 'quzihan2', 'shechanchen1', 'shechanchen2', 'tanyuanyong1', 'tanyuanyong2', 'wudi1',
            'wudi2', 'yepeng1', 'yepeng2', 'yumingwei1', 'yumingwei2', 'zhangboyu1', 'zhangboyu2', 'zhouxudong1',
            'zhouxudong2']

# 加载视频流
cap = cv2.VideoCapture("converge.mp4")

# 指定模型路径
model = load_model('model/restnet50_weights.h5')
# model = load_model('model/vgg16_weights.h5')
print("load model finished!")

while True:
    ret, img = cap.read()
    if not ret:
        break
    # copy原始图像
    or_img = copy.copy(img)
    resize_or_img = cv2.resize(or_img, (224, 224))

    # 将Opencv读取的BGR图像转换为RGB，参考https://blog.csdn.net/lly1122334/article/details/90239344
    img = PIL.Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 将图像resize到224 x 224分辨率
    img = img.resize((224, 224), PIL.Image.NEAREST)
    img = image.img_to_array(img) / 255.0
    # 扩维度 (224, 224, 3) -> (1, 224, 224, 3)
    img = np.expand_dims(img, axis=0)
    # 预测图片
    predictions = model.predict(img)
    # 获取预测标签
    predict_label = np.argmax(predictions[0])

    # 添加文字信息
    img = cv2.putText(img, name_lst[predict_label], (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    # 窗口偏移量
    cv2.moveWindow("video", 300, 300)
    cv2.imshow("video", resize_or_img)

    k = cv2.waitKey(10)
    if k == 27:  # ESC  END
        cap.release()
        cv2.destroyAllWindows()
        break
