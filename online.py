# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '/Users/jeff/ProjectCode/VideoSeg/ESD/online.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

import os
import cv2
import sys
import time
import time
import warnings
import argparse
import numpy as np
import pandas as pd
from threading import Thread
from datetime import datetime
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
from PyQt5.QtWidgets import QMainWindow, QFileDialog
from PyQt5.QtWidgets import QWidget, QPushButton, QDialog, QLabel

from utils.parser import ParserUse
from utils.guis import PhaseCom
from utils.report_tools import generate_report, get_meta

warnings.filterwarnings("ignore")

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    process_img_signal = pyqtSignal(np.ndarray, int)

    def run(self):
        frame_idx = 0
        # capture from web cam
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
        time.sleep(0.5)
        if cap.isOpened():
            while True:
                frame_idx += 1
                ret, cv_img = cap.read()
                # time.sleep(0.10)
                if ret:
                    self.change_pixmap_signal.emit(cv_img)
                    self.process_img_signal.emit(cv_img, frame_idx)
                else:
                    assert "Cannot get frame"
        else:
            cap.release()
            assert "Cannot get frame"

class Ui_iPhaser(QMainWindow):
    def __init__(self):
        super(Ui_iPhaser, self).__init__()

    def setupUi(self, cfg):
        self.setObjectName("iPhaser")
        self.resize(1300, 930)
        self.centralwidget = QWidget()

        self.disply_width = 1200
        self.display_height = 900
        self.save_folder = "../Records"
        if not os.path.isdir(self.save_folder):
            os.makedirs(self.save_folder)
        self.down_ratio = cfg.down_ratio
        self.start_time = "--:--:--"
        self.trainee_name = "--"
        self.manual_set = "--"

        # Statue parameters
        self.init_status()
        self.FRAME_WIDTH, self.FRAME_HEIGHT, self.stream_fps = self.get_frame_size()
        # self.FRAME_WIDTH = 1920
        # self.FRAME_HEIGHT = 1440
        # self.stream_fps = 50
        self.MANUAL_FRAMES = self.stream_fps * cfg.manual_set_fps_ratio
        self.manual_frame = 0
        # self.FRAME_WIDTH = 1280
        # self.FRAME_HEIGHT = 720
        self.CODEC = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')

        self.date_time = datetime.now().strftime("%d/%m/%Y-%H:%M:%S.%f")
        self.fps = 0
        self.pred = "--"
        self.log_data = []

        self.phaseseg = PhaseCom(arg=cfg)

        self.centralwidget = QtWidgets.QWidget()
        self.centralwidget.setObjectName("centralwidget")
        self.DisplayVideo = QtWidgets.QLabel(self.centralwidget)
        self.DisplayVideo.setGeometry(QtCore.QRect(50, 80, 1200, 900))
        self.DisplayVideo.setStyleSheet("background-color: rgb(197, 197, 197);")
        self.DisplayVideo.setText("")
        self.DisplayVideo.setObjectName("DisplayVideo")

        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(50, 10, 1200, 34))
        self.layoutWidget.setObjectName("layoutWidget")

        self.horizontalLayout = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")

        self.TraineeLabel = QtWidgets.QLabel(self.layoutWidget)
        self.TraineeLabel.setObjectName("TraineeLabel")
        self.horizontalLayout.addWidget(self.TraineeLabel)

        self.TraineeName = QtWidgets.QLineEdit(self.layoutWidget)
        self.TraineeName.setObjectName("TraineeName")
        self.horizontalLayout.addWidget(self.TraineeName)

        self.TrainerLabel = QtWidgets.QLabel(self.layoutWidget)
        self.TrainerLabel.setObjectName("TrainerLabel")
        self.horizontalLayout.addWidget(self.TrainerLabel)

        self.TrainerName = QtWidgets.QLineEdit(self.layoutWidget)
        self.TrainerName.setObjectName("TrainerName")
        self.horizontalLayout.addWidget(self.TrainerName)

        self.BedLabel = QtWidgets.QLabel(self.layoutWidget)
        self.BedLabel.setObjectName("BedLabel")
        self.horizontalLayout.addWidget(self.BedLabel)

        self.BedName = QtWidgets.QLineEdit(self.layoutWidget)
        self.BedName.setObjectName("BedName")
        self.horizontalLayout.addWidget(self.BedName)

        self.CaseLabel = QtWidgets.QLabel(self.layoutWidget)
        self.CaseLabel.setObjectName("CaseLabel")
        self.horizontalLayout.addWidget(self.CaseLabel)

        self.CaseName = QtWidgets.QLineEdit(self.layoutWidget)
        self.CaseName.setObjectName("CaseName")
        self.horizontalLayout.addWidget(self.CaseName)

        self.Start = QtWidgets.QPushButton(self.layoutWidget)
        self.Start.setObjectName("Start")
        self.Start.clicked.connect(self.click_start)
        self.horizontalLayout.addWidget(self.Start)

        self.Stop = QtWidgets.QPushButton(self.layoutWidget)
        self.Stop.setObjectName("Stop")
        self.Stop.clicked.connect(self.click_stop)
        self.horizontalLayout.addWidget(self.Stop)

        self.layoutWidget2 = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget2.setGeometry(QtCore.QRect(50, 34, 250, 40))
        self.layoutWidget2.setObjectName("layoutWidget2")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.layoutWidget2)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.TimeLabel = QtWidgets.QLabel(self.layoutWidget2)
        self.TimeLabel.setObjectName("TimeLabel")
        self.horizontalLayout_3.addWidget(self.TimeLabel)
        self.label = QtWidgets.QLabel(self.layoutWidget2)
        self.label.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.label.setText("--:--:--")
        self.label.setObjectName("label")
        self.horizontalLayout_3.addWidget(self.label)

        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(400, 34, 680, 40))
        self.widget.setObjectName("widget")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.ActionIndependent = QtWidgets.QPushButton(self.widget)
        self.ActionIndependent.setObjectName("ActionIndependent")
        self.ActionIndependent.clicked.connect(self.click_independent)
        self.horizontalLayout_4.addWidget(self.ActionIndependent)
        self.ActionHelp = QtWidgets.QPushButton(self.widget)
        self.ActionHelp.setObjectName("ActionHelp")
        self.ActionHelp.clicked.connect(self.click_help)
        self.horizontalLayout_4.addWidget(self.ActionHelp)

        self.ActionTakeOver = QtWidgets.QPushButton(self.widget)
        self.ActionTakeOver.setObjectName("ActionTakeOver")
        self.ActionTakeOver.clicked.connect(self.click_take_over)
        self.horizontalLayout_4.addWidget(self.ActionTakeOver)

        self.Report = QtWidgets.QPushButton(self.widget)
        self.Report.setObjectName("Report")
        self.Report.clicked.connect(self.click_report)
        self.horizontalLayout_4.addWidget(self.Report)

        # self.layoutWidget_report = QtWidgets.QWidget(self.centralwidget)
        # self.layoutWidget_report.setGeometry(QtCore.QRect(50, 80, 685, 34))
        # self.layoutWidget_report.setObjectName("layoutWidgetReport")
        #
        # self.reportLayout = QtWidgets.QHBoxLayout(self.layoutWidget_report)
        # self.reportLayout.setContentsMargins(0, 0, 0, 0)
        # self.reportLayout.setObjectName("reportLayout")
        #
        # self.ReportFileLabel = QtWidgets.QLabel(self.layoutWidget_report)
        # self.ReportFileLabel.setObjectName("ReportFileLabel")
        # self.reportLayout.addWidget(self.ReportFileLabel)
        #
        # self.ReportFile = QtWidgets.QLineEdit(self.layoutWidget_report)
        # self.ReportFile.setObjectName("ReportFile")
        # self.reportLayout.addWidget(self.ReportFile)
        #
        # self.ChooseReport = QtWidgets.QPushButton(self.layoutWidget_report)
        # self.ChooseReport.setObjectName("ChooseReport")
        # self.ChooseReport.clicked.connect(self.click_choose_report)
        # self.reportLayout.addWidget(self.ChooseReport)
        #
        # self.Report = QtWidgets.QPushButton(self.layoutWidget_report)
        # self.Report.setObjectName("Report")
        # self.Report.clicked.connect(self.click_report)
        # self.reportLayout.addWidget(self.Report)


        self.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(self)
        self.statusbar.setObjectName("statusbar")
        self.setStatusBar(self.statusbar)

        self.retranslateUi()
        QtCore.QMetaObject.connectSlotsByName(self)

        # create the video capture thread
        self.thread = VideoThread()
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.process_img_signal.connect(self.process_img)
        # start the thread
        self.thread.start()

    def init_status(self):
        self.WORKING = False
        self.INIT = False
        self.TRAINEE = "NONE"
        self.PAUSE_times = 0
        self.INDEPENDENT = True
        self.HELP = False
        self.STATUS = "--"

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("iPhaser", "iPhaser"))
        self.TraineeLabel.setText(_translate("iPhaser", "Trainee:"))
        self.TrainerLabel.setText(_translate("iPhaser", "Mentor:"))
        self.BedLabel.setText(_translate("iPhaser", "Bed:"))
        self.CaseLabel.setText(_translate("iPhaser", "Case:"))
        self.Start.setText(_translate("iPhaser", "Start"))
        self.Stop.setText(_translate("iPhaser", "Stop"))
        # self.ReportFileLabel.setText(_translate("iPhaser", "Report file:"))
        # self.ChooseReport.setText(_translate("iPhaser", "Choose"))
        self.Report.setText(_translate("iPhaser", "> Generate report <"))
        # self.CurrentTrainee.setText(_translate("iPhaser", "<html><head/><body><p><span style=\" font-size:18pt;\">Current Trainee:</span></p></body></html>"))
        # self.PhaseLabel.setText(_translate("iPhaser", "<html><head/><body><p><span style=\" font-size:18pt; font-weight:600;\">Predicted phase:</span></p></body></html>"))
        self.TimeLabel.setText(_translate("iPhaser", "Time start:"))
        self.ActionIndependent.setText(_translate("iPhaser", "Independent"))
        self.ActionHelp.setText(_translate("iPhaser", "With help"))
        self.ActionTakeOver.setText(_translate("iPhaser", "Take over"))

    # Buttons
    def click_start(self):
        self.WORKING = True
        self.log_data = []
        self.Start.setEnabled(False)
        self.Stop.setEnabled(True)
        self.ActionIndependent.setEnabled(True)
        self.ActionHelp.setEnabled(True)
        self.ActionTakeOver.setEnabled(True)
        self.start_time = datetime.now().strftime("%H:%M:%S")
        self.label.setText(self.start_time)
        self.trainee_name = self.TraineeName.text()
        self.trainer_name = self.TrainerName.text()
        self.bed_name = self.BedName.text()
        self.case_name = self.CaseName.text()
        self.TraineeName.setEnabled(False)
        self.TrainerName.setEnabled(False)
        self.BedName.setEnabled(False)
        self.CaseName.setEnabled(False)
        self.log_file = os.path.join(self.save_folder, self.case_name + "_" + self.trainee_name + "_" + self.start_time.replace(":", "-") + ".csv")
        video_file_name = os.path.join(self.save_folder, self.case_name + "_" + self.trainee_name + "_" + self.start_time.replace(":", "-") + ".avi")
        self.output_video = cv2.VideoWriter(video_file_name, self.CODEC, self.stream_fps, (self.FRAME_WIDTH, self.FRAME_HEIGHT))
        # self.DisplayTrainee.setText(self.trainee_name)

    def click_stop(self):
        self.WORKING = False
        self.save_log_data()
        self.label.setText("--:--:--")
        self.init_status()
        self.output_video.release()
        self.TraineeName.setEnabled(True)
        self.TrainerName.setEnabled(True)
        self.BedName.setEnabled(True)
        self.CaseName.setEnabled(True)
        self.Start.setEnabled(True)
        self.Stop.setEnabled(False)
        # self.DisplayTrainee.setText("--")

    def click_choose_report(self):
        fileName_choose, filetype = QFileDialog.getOpenFileName(self,
                                                                "选取文件",
                                                                os.getcwd(),  # Default dir
                                                                "All Files (*);;Text Files (*.csv)")  # 设置文件扩展名过滤,用双分号间隔

        self.reportfile = fileName_choose
        self.ReportFile.setText(fileName_choose)

    def click_report(self):
        self.Report.setEnabled(False)
        report_path = generate_report("../Records")
        fullpath = os.path.realpath("./reports")
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(fullpath))

        self.Report.setEnabled(True)

    def click_independent(self):
        self.INDEPENDENT = True
        self.INIT = True
        self.STATUS = "Indepedent"
        # self.log_data.append(["Independent"] * 5)
        self.ActionIndependent.setEnabled(False)
        self.ActionHelp.setEnabled(True)
        self.ActionTakeOver.setEnabled(True)

    def click_help(self):
        # self.log_data.append(["Help"] * 5)
        self.INIT = True
        self.STATUS = "Help"
        self.ActionIndependent.setEnabled(True)
        self.ActionHelp.setEnabled(False)
        self.ActionTakeOver.setEnabled(True)

    def click_take_over(self):
        self.INIT = True
        self.STATUS = "TakeOver"
        # self.log_data.append(["TakeOver"] * 5)
        self.ActionIndependent.setEnabled(True)
        self.ActionHelp.setEnabled(True)
        self.ActionTakeOver.setEnabled(False)

    def save_log_data(self):
        datas = zip(*self.log_data)
        data_dict = {}
        names = ["Time", "Frame", "Trainee", "Trainer", "Bed", "Status", "FPS", "Prediction", "Correction"]
        for name, data in zip(names, datas):
            data_dict[name] = list(data)
        pd_log = pd.DataFrame.from_dict(data_dict)
        preds = pd_log["Prediction"].tolist()
        correcs = pd_log["Correction"].tolist()
        combines = []
        for pred, corr in zip(preds, correcs):
            if corr == "--":
                combines.append(pred)
            else:
                combines.append(corr)
        pd_log["Combine"] = combines
        curent_date_time = "_" + datetime.now().strftime("%H-%M-%S") + ".csv"
        pd_log.to_csv(self.log_file.replace(".csv", curent_date_time), index=False, header=True)

    def get_frame_size(self):
        capture = cv2.VideoCapture(0)

        # Default resolutions of the frame are obtained (system dependent)
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(capture.get(cv2.CAP_PROP_FPS))
        capture.release()
        return frame_width, frame_height, fps

    def process_img(self, cv_img, frame_idx):
        cv_img = cv_img[30:1050, 695:1850]
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        if frame_idx % self.down_ratio == 0 and self.WORKING:
            self.date_time = datetime.now().strftime("%d/%m/%Y-%H:%M:%S.%f")
            start_time = time.time()
            self.pred = self.phaseseg.seg_frame(rgb_image)
            # self.DisplayPhase.setText(self.pred)
            end_time = time.time()
            self.fps = 1/np.round(end_time - start_time, 3)

            # add log data
            self.log_data.append([self.date_time, str(frame_idx).zfill(7), self.trainee_name, self.trainer_name, self.bed_name, self.STATUS, "{:>7.4f}".format(self.fps), self.pred, self.manual_set])

    def keyPressEvent(self, e):
        pressed_key = e.text()
        if pressed_key == "a":
            self.manual_frame = self.MANUAL_FRAMES
            self.manual_set = "idle"
        elif pressed_key == "s":
            self.manual_frame = self.MANUAL_FRAMES
            self.manual_set = "marking"
        elif pressed_key == "d":
            self.manual_frame = self.MANUAL_FRAMES
            self.manual_set = "injection"
        elif pressed_key == "f":
            self.manual_frame = self.MANUAL_FRAMES
            self.manual_set = "dissection"

    def update_image(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        # Collect settings of functional keys
        # cv_img = cv_img[30:1050, 695:1850]
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        self.manual_frame = self.manual_frame - 1
        if self.manual_frame <= 0:
            self.manual_frame = 0
            self.manual_set = "--"
        if self.INIT:
            self.date_time = datetime.now().strftime("%d/%m/%Y-%H:%M:%S")
            if self.manual_frame > 0:
                self.pred = self.manual_set
            rgb_image = self.phaseseg.add_text(self.date_time, self.pred, self.trainee_name, rgb_image)
        if self.WORKING:
            self.output_video.write(rgb_image)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        p = QPixmap.fromImage(p)
        self.DisplayVideo.setPixmap(p)

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("-s", default=False,  action='store_true', help="Whether save predictions")
    parse.add_argument("-q", default=False, action='store_true', help="Display video")
    parse.add_argument("--cfg", default="test", type=str)

    cfg = parse.parse_args()
    cfg = ParserUse(cfg.cfg, "camera").add_args(cfg)

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    ui = Ui_iPhaser()
    ui.setupUi(cfg)
    ui.show()
    sys.exit(app.exec_())

