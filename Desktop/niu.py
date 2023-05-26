'''
Created on 2023年5月15日
@author：niuyong
@description:本程序用于进行人脸身份识别
@version 2.0
@CopyRight:CQUT
'''
from PyQt5.QtWidgets import QMainWindow, QHBoxLayout, QPushButton, QApplication, QWidget
import sys

import os

import cv2 as cv
import numpy as np

from GUI import Ui_Dialog

face_detector = cv.CascadeClassifier('haarcascade_frontalface_alt2.xml')  # 加载分类器
recognizer = cv.face.LBPHFaceRecognizer_create()  # 生成LBPH识别器实例模型


class MyWindow(QMainWindow, Ui_Dialog):
    def __init__(self, parent=None):
        super(MyWindow, self).__init__(parent)
        self.setupUi(self)
        self.pushButton.clicked.connect(self.record_train_data)
        self.pushButton_2.clicked.connect(self.train)
        self.pushButton_3.clicked.connect(self.predict)

    def record_train_data(self):
        capture = cv.VideoCapture(0)  # 打开摄像头
        # capture = cv.VideoCapture('peng.mp4')
        count = 0
        face_id = input('输入保存数据的名称:')
        while True:
            ret, frame = capture.read()  # 截取一帧图片
            if not ret:
                break
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # 灰度化
            faces = face_detector.detectMultiScale(frame, 1.3, 5)  # 检测人脸
            if len(faces) == 1:
                for (x, y, w, h) in faces:
                    cv.rectangle(frame, (x, y), (x + w, y + w), (255, 0, 0))  # 画出人脸
                    count += 1
                    cv.imwrite("data/User-" + str(face_id) + '-' + str(count) + '.jpg', gray[y:y + h, x:x + w])
            # 显示图片
            cv.imshow('image', frame)
            k = cv.waitKey(1)
            if k == 27:  # 按ESC或者照片收集到800张结束
                break
            elif count >= 800:
                break
        # 关闭摄像头，释放资源
        capture.release()
        cv.destroyAllWindows()

    def create_dataset(self):
        imgs = []
        ids = []
        root = './data'
        names = os.listdir(root)
        for name in names:
            suffix = name.split('.')[-1]
            info = name.split('.')[0]
            user_id = int(info.split('-')[1])
            img = cv.imread(f'{root}/{name}', 0)
            ids.append(user_id)
            imgs.append(img)
        return imgs, ids

    def train(self):
        save_dir = './trainer'
        faces, ids = self.create_dataset()
        print('training data......')
        recognizer.train(faces, np.array(ids))  # 对每个参考图像计算LBPH ,得到一个向量。每个人脸都是整个向量集中的一个点
        recognizer.save(f'{save_dir}/trainer.yml')

    def predict(self):
        recognizer.read('./trainer/trainer.yml')
        capture = cv.VideoCapture(0)
        # capture = cv.VideoCapture('./test_demo.mp4')
        while True:
            ret, frame = capture.read()
            if not ret:
                break
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(frame, 1.3, 5)
            for (x, y, w, h) in faces:
                cv.rectangle(frame, (x, y), (x + w, y + w), (255, 0, 0))
                user_id, score = recognizer.predict(
                    gray[y:y + h, x:x + w])  # 对一个待测人脸图像进行判断,寻找与当前图像距离最近的人脸图像。与哪个人脸图像最近,就将当前待测图像标注为其对应的标签。
                print(f'user{user_id} --- score: {score}')
                if score < 50:  # 置信度小于50则认为可以接受
                    cv.putText(frame, f'{user_id}', (x + 10, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
                else:
                    cv.putText(frame, 'unkonw', (x + 10, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
            # 显示图片
            cv.imshow('image', frame)
            k = cv.waitKey(24)
            if k == 27:
                break
        # 关闭摄像头，释放资源
        capture.release()
        cv.destroyAllWindows()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = MyWindow()
    myWin.show()
    sys.exit(app.exec_())
