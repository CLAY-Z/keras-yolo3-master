# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'hat_recognization.ui'
#
# Created by: PyQt5 UI code generator 5.13.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5.QtGui import QMovie, QFont
from PyQt5.QtWidgets import QFileDialog
from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
from os import getcwd, remove
from slice_png import img as bgImg
from base64 import b64decode
import time
import numpy as np
from yolo_model_ui import YOLO
from PIL import Image
import os


class Ui_MainWindow(object):
    # 初始化ui窗口
    def __init__(self, MainWindow):
        self.class_names = ['hat', 'person']

        self.path = getcwd()
        self.timer_camera = QtCore.QTimer()  # 定时器

        self.setupUi(MainWindow)
        self.retranslateUi(MainWindow)
        self.slot_init()
        self.image = None

        # 字体大小
        self.ft = QFont()
        self.ft.setPointSize(16)

        # 设置界面动画
        gif = QMovie('./images_ui/scan.gif')
        gif.setScaledSize(QtCore.QSize(420, 280))
        self.label_show.setMovie(gif)
        gif.start()

        self.cap = cv2.VideoCapture()  # 屏幕画面对象
        self.CAM_NUM = 0  # 摄像头标号

        self.url = "rtsp://admin:HK123456@192.168.1.20:8000/Streaming/Channels/1"  # 网络摄像头协议

        self.model_path = 'logs/trained_weights'  # 模型路径

        self.model = None
        self.outdir = "C:/Users/zhaoxiaohe/Desktop/image"

        self.initUI()

    def initUI(self):
        self.textEdit_video.setText('请选择识别视频')
        self.textEdit_camera.setText('请选择摄像头（默认号：0）')
        self.textEdit_image.setText('请选择识别图片')
        self.textEdit_model.setText('请选择模型（默认YoloV3）')
        self.label_result_2.clear()
        self.label_result_2.setStyleSheet("border-image: url(./images_ui/slice.png);")

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(803, 609)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.toolButton_camera = QtWidgets.QToolButton(self.centralwidget)
        self.toolButton_camera.setGeometry(QtCore.QRect(90, 110, 81, 31))
        self.toolButton_camera.setObjectName("toolButton_camera")
        self.textEdit_camera = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_camera.setGeometry(QtCore.QRect(190, 110, 231, 31))
        self.textEdit_camera.setObjectName("textEdit_camera")
        self.toolButton_image = QtWidgets.QToolButton(self.centralwidget)
        self.toolButton_image.setGeometry(QtCore.QRect(90, 160, 81, 31))
        self.toolButton_image.setObjectName("toolButton_image")
        self.textEdit_image = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_image.setGeometry(QtCore.QRect(190, 160, 231, 31))
        self.textEdit_image.setObjectName("textEdit_image")
        self.toolButton_model = QtWidgets.QToolButton(self.centralwidget)
        self.toolButton_model.setGeometry(QtCore.QRect(90, 210, 81, 31))
        self.toolButton_model.setObjectName("toolButton_model")
        self.textEdit_model = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_model.setGeometry(QtCore.QRect(190, 210, 231, 31))
        self.textEdit_model.setObjectName("textEdit_model")
        self.pushButton_start = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_start.setGeometry(QtCore.QRect(560, 80, 121, 61))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(14)
        self.pushButton_start.setFont(font)
        self.pushButton_start.setObjectName("pushButton_start")
        self.pushButton_end = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_end.setGeometry(QtCore.QRect(560, 160, 121, 61))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(14)
        self.pushButton_end.setFont(font)
        self.pushButton_end.setObjectName("pushButton_end")
        self.label_show = QtWidgets.QLabel(self.centralwidget)
        self.label_show.setGeometry(QtCore.QRect(60, 270, 420, 280))
        self.label_show.setMinimumSize(QtCore.QSize(420, 280))
        self.label_show.setMaximumSize(QtCore.QSize(420, 280))
        self.label_show.setFrameShape(QtWidgets.QFrame.Box)
        self.label_show.setText("")
        self.label_show.setObjectName("label_show")
        self.label_result_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_result_2.setGeometry(QtCore.QRect(520, 310, 250, 230))
        self.label_result_2.setFrameShape(QtWidgets.QFrame.Box)
        self.label_result_2.setText("")
        self.label_result_2.setObjectName("label_result_2")
        self.label_useTime = QtWidgets.QLabel(self.centralwidget)
        self.label_useTime.setGeometry(QtCore.QRect(540, 240, 81, 41))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(14)
        self.label_useTime.setFont(font)
        self.label_useTime.setObjectName("label_useTime")
        self.label_time = QtWidgets.QLabel(self.centralwidget)
        self.label_time.setGeometry(QtCore.QRect(640, 240, 91, 41))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(14)
        self.label_time.setFont(font)
        self.label_time.setTextFormat(QtCore.Qt.AutoText)
        self.label_time.setObjectName("label_time")
        self.label_scanResult = QtWidgets.QLabel(self.centralwidget)
        self.label_scanResult.setGeometry(QtCore.QRect(540, 280, 91, 31))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(14)
        self.label_scanResult.setFont(font)
        self.label_scanResult.setObjectName("label_scanResult")
        self.label_result = QtWidgets.QLabel(self.centralwidget)
        self.label_result.setGeometry(QtCore.QRect(630, 270, 140, 41))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(16)
        self.label_result.setFont(font)
        self.label_result.setStyleSheet("color: rgb(0, 189, 189);")
        self.label_result.setObjectName("label_result")
        self.toolButton_video = QtWidgets.QToolButton(self.centralwidget)
        self.toolButton_video.setGeometry(QtCore.QRect(90, 60, 81, 31))
        self.toolButton_video.setObjectName("toolButton_video")
        self.textEdit_video = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_video.setGeometry(QtCore.QRect(190, 60, 231, 31))
        self.textEdit_video.setObjectName("textEdit_video")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "施工现场危险行为监控系统v0.1"))
        self.toolButton_camera.setText(_translate("MainWindow", "打开摄像头"))
        self.toolButton_image.setText(_translate("MainWindow", "选择图片"))
        self.toolButton_model.setText(_translate("MainWindow", "选择模型"))
        self.pushButton_start.setText(_translate("MainWindow", "运行"))
        self.pushButton_end.setText(_translate("MainWindow", "退出"))
        self.label_useTime.setText(_translate("MainWindow", "<html><head/><body><p>所用时间：</p></body></html>"))
        self.label_time.setText(_translate("MainWindow", "0 s"))
        self.label_scanResult.setText(_translate("MainWindow", "<html><head/><body><p>目前状态：<br/></p></body></html>"))
        self.label_result.setText(_translate("MainWindow", "未知"))
        self.toolButton_video.setText(_translate("MainWindow", "选择视频"))

    def slot_init(self):  # 定义槽函数
        self.toolButton_camera.clicked.connect(self.button_open_camera_click)
        self.toolButton_model.clicked.connect(self.choose_model)
        self.toolButton_image.clicked.connect(self.choose_picture)
        self.toolButton_video.clicked.connect(self.button_open_video_click)
        self.timer_camera.timeout.connect(self.show_camera)
        self.pushButton_end.clicked.connect(self.closeWindow)

    def button_open_video_click(self):
        if not self.timer_camera.isActive():
            # 调用文件选择对话框
            filename_choose, filetype = QFileDialog.getOpenFileName(self.centralwidget,
                                                                    '选取视频文件',
                                                                    getcwd(),
                                                                    'video File (*.mp4; *.avi)')
            # self.cap = cv2.VideoCapture()
            flag = self.cap.open(filename_choose)
            if not flag:  # 视频打开失败提醒
                msg = QtWidgets.QMessageBox.warning(self.centralwidget, u"Warning", u"请检查视频文件是否存在、格式是否正确，以及是否损坏！",
                                                    buttons=QtWidgets.QMessageBox.Ok,
                                                    defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                # 准备运行识别程序
                self.textEdit_image.setText("图片未选中")
                self.textEdit_image.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
                self.textEdit_camera.setText("实时摄像头未开启")
                self.textEdit_camera.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
                self.textEdit_video.setText(os.path.basename(filename_choose) + '视频已选中')
                self.textEdit_video.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
                self.label_show.setText("正在启动识别系统...\n\nloading...")
                self.label_show.setFont(self.ft)
                self.label_show.setStyleSheet("color: red")
                self.label_show.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)  # 文本居中显示

                # 生成模型对象
                self.model = YOLO()
                QtWidgets.QApplication.processEvents()
                # 打开定时器,定时30ms,1s = 1000ms
                self.timer_camera.start(30)

        else:
            # 定时器未开启，界面回复初始状态
            self.timer_camera.stop()
            self.cap.release()
            self.label_show.clear()
            self.textEdit_camera.setText('实时摄像已关闭')
            self.textEdit_image.setText('图片未选中')
            self.label_result_2.clear()
            self.label_result_2.setStyleSheet("border-image: url(./images_ui/ini.png);")
            self.label_result.setText('未知')
            self.label_time.setText('0 s')

    def button_open_camera_click(self):
        if not self.timer_camera.isActive():  # isActive():检查定时状态，如果定时器正在运行，返回真，否则返回假
            self.cap = cv2.VideoCapture()
            flag = self.cap.open(self.url)  # 检查摄像头状态
            if not flag:  # 相机打开失败提醒
                msg = QtWidgets.QMessageBox.warning(self.centralwidget, u"Warning", u"请检测摄像头与电脑是否连接！",
                                                    buttons=QtWidgets.QMessageBox.Ok,
                                                    defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                # 准备运行识别程序
                self.textEdit_image.setText("图片未选中")
                QtWidgets.QApplication.processEvents()
                self.textEdit_camera.setText("实时摄像头已开启")
                self.label_show.setText("正在启动识别系统...\n\nloading...")
                self.label_show.setFont(self.ft)
                self.label_show.setStyleSheet("color: red")
                self.label_show.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)  # 文本居中显示

                QtWidgets.QApplication.processEvents()
                # 打开定时器,定时30ms,1s = 1000ms
                self.timer_camera.start(30)  # 该语句不是只运行一次，而是一直运行下去，所以需要一个判断语句来结束
        else:
            # 定时器未开启，界面回复初始状态
            self.timer_camera.stop()
            self.cap.release()
            self.label_show.clear()
            self.textEdit_camera.setText('实时摄像已关闭')
            self.textEdit_image.setText('图片未选中')
            self.label_result_2.clear()
            self.label_result_2.setStyleSheet("border-image: url(./images_ui/ini.png);")

            self.label_result.setText('未知')
            self.label_time.setText('0 s')

    def choose_model(self):
        # 选择训练好的模型
        self.timer_camera.stop()
        # self.cap.release()
        self.label_show.clear()
        self.label_result.setText('未知')
        self.label_time.setText('0 s')
        self.textEdit_camera.setText('实时摄像头已关闭')
        self.label_result_2.clear()
        self.label_result_2.setStyleSheet("border-image: url(./images_ui/ini.png);")

        # 调用文件选择对话框
        filename_choose, filetype = QFileDialog.getOpenFileName(self.centralwidget,
                                                                '选取模型文件',
                                                                getcwd(),
                                                                'Model File (.h5)')
        # 显示提示信息
        if filename_choose != '':
            self.model_path = filename_choose
            self.textEdit_model.setText(filename_choose + '已选中')
        else:
            self.textEdit_model.setText('使用默认模型')

        # 恢复界面
        gif = QMovie('./images_ui/scan.gif')
        self.label_show.setMovie(gif)
        gif.start()

    # 定时器超过30ms后，发送信号至timeout()，后者再与槽函数show_camera相连，触发槽函数，即30ms调用一次槽函数
    def show_camera(self):
        # 定时器槽函数，每隔30ms时间执行
        flag, self.image = self.cap.read()  # 获取一帧图像
        print(type(self.image))
        self.image = cv2.flip(self.image, 1)  # 左右翻转

        # 设置数据显示的背景
        tmp = open('slice.png', 'wb')
        tmp.write(b64decode(bgImg))
        tmp.close()
        canvas = cv2.imread('slice.png')
        remove('slice.png')  # 删除指定文件

        # 开始使用模型预测图片
        time_start = time.time()  # 计时
        self.image = Image.fromarray(self.image.astype('uint8'))
        print(type(self.image))
        image, out_classes, out_scores, out_boxes = self.model.detect_image(self.image)  # 返回的也是一个JpegImageFile对象
        time_end = time.time()

        # 显示结果
        img_arr = np.array(image)
        img_arr = cv2.resize(img_arr, (420, 280))
        show = cv2.cvtColor(img_arr, cv2.COLOR_BGR2BGRA)  # (300, 450, 3) (h w c)
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB32)
        image_set = QtGui.QPixmap.fromImage(showImage)
        self.label_show.setPixmap(image_set)

        self.label_result.setText('安全')
        for i, c in reversed(list(enumerate(out_classes))):
            # 提取每个目标的类别、框以及得分信息
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            if predicted_class == 'person':
                self.label_result.setText('未佩戴安全帽')

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

            self.label_result_2.setAlignment(QtCore.Qt.AlignCenter)
            self.label_result_2.setText('共发现 {} 个识别目标，其中：\n\n\n 类别：{},\n\n 置信度：{:.2f},\n\n 目标位置：({}, {}, {}, {})'
                                        '\n\n'.format(len(out_boxes), predicted_class, score, top, left, bottom, right))

        # self.label_result_2.setText(result)
        self.label_time.setText(str(round((time_end - time_start), 3)) + 's')

    def choose_picture(self):
        # 界面处理
        self.timer_camera.stop()
        # self.cap.release()
        self.label_show.clear()
        self.label_result.setText('未知')
        self.label_time.setText('0 s')
        self.textEdit_camera.setText('实时摄像头已关闭')
        self.label_show.setStyleSheet("color: red;background-color: rgb(162, 200, 198);")

        # filename_choose为图片的绝对路径
        filename_choose, filetype = QFileDialog.getOpenFileName(self.centralwidget,
                                                                '选取图片文件',
                                                                self.path,
                                                                '图片(*.jpg; *.jpeg; *.png)')
        # self.path = filename_choose  # 保存路径
        if filename_choose != '':
            self.textEdit_image.setText(os.path.basename(filename_choose) + '文件已选中')
            self.label_show.setText('正在启动识别系统\n\n请稍等...')
            self.label_show.setFont(self.ft)

            self.label_show.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)  # 文本居中显示

            QtWidgets.QApplication.processEvents()  # 多线程进行处理

            # 生成模型对象
            self.model = YOLO()
            image = Image.open(filename_choose)  # 读取选择的图片，返回一个JpegImageFile对象

            # 计时并开始模型预测
            QtWidgets.QApplication.processEvents()
            time_start = time.time()
            image, out_classes, out_scores, out_boxes = self.model.detect_image(image)  # 返回的也是一个JpegImageFile对象
            image.save(os.path.join(self.outdir, os.path.basename(filename_choose)))
            time_end = time.time()

            # 显示图片结果
            img_arr = np.array(image)
            img_arr = cv2.resize(img_arr, (420, 280))
            show = cv2.cvtColor(img_arr, cv2.COLOR_BGR2BGRA)  # (300, 450, 3) (h w c)
            # show.data = <memory at 0x000001F15B577228>
            showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB32)
            images = showImage.rgbSwapped()  # RGB->BGR
            image_set = QtGui.QPixmap.fromImage(images)
            self.label_show.setPixmap(image_set)

            # # 在显示结果的label中显示结果
            # show = cv2.cvtColor(canvas, cv2.COLOR_BGR2BGRA)
            # showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB32)
            # self.label_result_2.setPixmap(QtGui.QPixmap.fromImage(showImage))

            # 显示结果
            self.label_result.setText('安全')
            for i, c in reversed(list(enumerate(out_classes))):
                # 提取每个目标的类别、框以及得分信息
                predicted_class = self.class_names[c]
                box = out_boxes[i]
                score = out_scores[i]

                if predicted_class == 'person':
                    self.label_result.setText('未佩戴安全帽')

                top, left, bottom, right = box
                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
                right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

                self.label_result_2.setAlignment(QtCore.Qt.AlignCenter)
                self.label_result_2.setText('共发现 {} 个识别目标，其中：\n\n\n 类别：{},\n\n 置信度：{:.2f},\n\n 目标位置：({}, {}, {}, {})'
                                            '\n\n'.format(len(out_boxes), predicted_class, score, top, left, bottom, right))

            self.label_time.setText(str(round((time_end - time_start), 3)) + 's')

        else:
            # 选择取消，恢复界面状态
            self.textEdit_image.setText('文件未选中')
            gif = QMovie('./images_ui/scan.gif')
            self.label_show.setMovie(gif)
            gif.start()
            self.label_result_2.clear()  # 清除画面
            self.label_result_2.setStyleSheet('border-image: url(./images_ui/ini.png);')
            self.label_result.setText('None')
            self.label_time.setText('0 s')

    def cv_imread(self, filePath):
        # 读取图片
        cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), -1)
        # imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
        # cv_img=cv2.cvtColor(cv_img,cv2.COLOR_RGB2BGR)
        return cv_img

    def closeWindow(self):
        reply = QtWidgets.QMessageBox.question(self.centralwidget,
                                               '警告',
                                               '退出后监测将终止,\n确认要退出吗？',
                                               QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            QtCore.QCoreApplication.quit()
