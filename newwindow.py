from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import *
import os
import sys
from pathlib import Path
import threading
from threading import Thread
from PyQt5.QtCore import Qt
from newui import res_rc
import cv2
import torch
import torch.backends.cudnn as cudnn
from PyQt5.QtWidgets import QFileDialog

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadStreams
from utils.general import apply_classifier, check_img_size, check_imshow, check_requirements, check_suffix, colorstr, \
    increment_path, non_max_suppression, print_args, save_one_box, scale_coords, set_logging, \
    strip_optimizer, xyxy2xywh
from utils.plots import Annotator, colors
from utils.torch_utils import load_classifier, select_device, time_sync


@torch.no_grad()
class Ui_MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__()
        self.device = '0'
        self.weights = "pretrained/best.pt"
        self.source = " "
        self.imgsz = 640
        self.half = False
        self.names = None
        self.webcam = False
        self.camflag = False
        self.cap = None
        self.stride = None
        self.model = None
        self.modelc = None
        self.pt = None
        self.onnx = None
        self.tflite = None
        self.pb = None
        self.saved_model = None

        # Initialize
        set_logging()
        self.setupUi(self)
        self.setWindowIcon(QIcon("UI/icon.png"))
        # 初始化Qlabel界面
        self.initLogo()
        # 初始化点击触发函数
        self.initSlots()
        # 初始化权重
        self.initWeight()
        self.stopEvent = threading.Event()
        # 隐藏多余的窗体
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        # 展示
        self.show()

    # 更新按钮状态
    def update_button_states(self, state):
        self.pushButton_Photo.setEnabled(state)
        self.pushButton_Carera.setEnabled(state)
        self.pushButton_Video.setText("Video")

    # 点击触发函数
    def initSlots(self):
        self.pushButton_Photo.clicked.connect(self.button_photo_open)
        self.pushButton_Video.clicked.connect(self.button_video_open)
        self.pushButton_Carera.clicked.connect(self.button_camera_open)

    # 初始化权重
    def initWeight(self):
        self.device = select_device(self.device)
        self.half &= self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        w = str(self.weights[0] if isinstance(self.weights, list) else self.weights)
        classify, suffix, suffixes = False, Path(w).suffix.lower(), ['.pt', '.onnx', '.tflite', '.pb', '']
        check_suffix(w, suffixes)  # check weights have acceptable suffix
        self.pt, self.onnx, self.tflite, self.pb, self.saved_model = (suffix == x for x in suffixes)  # backend booleans
        if self.pt:
            self.model = torch.jit.load(w) if 'torchscript' in w else attempt_load(self.weights,
                                                                                   map_location=self.device)
            self.stride = int(self.model.stride.max())  # model stride
            self.names = self.model.module.names if hasattr(self.model,
                                                            'module') else self.model.names  # get class names
            if self.half:
                self.model.half()  # to FP16
            if classify:  # second-stage classifier
                self.modelc = load_classifier(name='resnet50', n=2)  # initialize
                self.modelc.load_state_dict(torch.load('resnet50.pt', map_location=self.device)['model']).to(
                    self.device).eval()
        else:
            print("No weights")

    # 初始化Qlabel界面
    def initLogo(self):
        pix = QtGui.QPixmap('UI/yolo.png')
        self.label.setScaledContents(True)
        self.label.setPixmap(pix.scaled(765, 595, Qt.KeepAspectRatio))
        print("initLogo")

    def button_photo_open(self):
        print('button_photo_open')
        img_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择图片", "",
                                                            "Image Files (*.png *.jpg *.bmp *.jpeg *.gif *.tif)")
        if not img_name:
            return
        thread1 = Thread(target=self.run, kwargs={"weights": self.weights, "source": str(img_name), "nosave": True,
                                                  "view_img": True})
        thread1.start()

    def button_video_open(self):
        print('button_video_open')
        if self.pushButton_Video.text() == "Stop":
            self.stopEvent.set()  # 设置停止信号
            self.pushButton_Video.setText("Video")
            self.pushButton_Photo.setEnabled(True)
            self.pushButton_Carera.setEnabled(True)
        else:
            video_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择视频", "",
                                                                  "Video Files (*.mp4 *.avi *.mov *.mkv *.flv *.wmv)")
            if not video_name:
                return
            if video_name.endswith("mp4"):
                self.pushButton_Video.setText("Stop")
                self.pushButton_Photo.setEnabled(False)
                self.pushButton_Carera.setEnabled(False)
                self.run_video(video_name)  # 直接调用视频识别函数

    def run_video(self, video_path):
        self.stopEvent.clear()  # 清除停止信号
        thread1 = Thread(target=self.run, kwargs={"weights": self.weights, "source": str(video_path), "nosave": True,
                                                  "view_img": True})
        thread1.start()

    def button_camera_open(self):
        if self.camflag == False:
            print('button_camera_open')
            self.webcam = True
            self.pushButton_Photo.setEnabled(False)
            self.pushButton_Video.setEnabled(False)
            self.run_camera()
        else:
            print('button_camera_close')
            self.stopEvent.set()  # 设置停止信号
            self.camflag = False
            self.webcam = False
            self.pushButton_Photo.setEnabled(True)
            self.pushButton_Video.setEnabled(True)
            self.pushButton_Carera.setText("Camera")
            self.initLogo()

    def run_camera(self):
        self.stopEvent.clear()  # 清除停止信号
        thread2 = Thread(target=self.run,
                         kwargs={"weights": self.weights, "source": "0", "nosave": True, "view_img": True})
        thread2.start()

    def setupUi(self, InterfaceWindow):
        InterfaceWindow.setObjectName("InterfaceWindow")
        InterfaceWindow.resize(1041, 700)
        self.centralwidget = QtWidgets.QWidget(InterfaceWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(19, 19, 1001, 661))
        self.frame.setMinimumSize(QtCore.QSize(1001, 661))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.frame)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.frame_2 = QtWidgets.QFrame(self.frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.frame_2.sizePolicy().hasHeightForWidth())
        self.frame_2.setSizePolicy(sizePolicy)
        self.frame_2.setStyleSheet("background-color: rgb(255, 255, 255);\n"
                                   "border-top-left-radius:20px;\n"
                                   "border-top-right-radius:20px;\n"
                                   "")
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.frame_2)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.frame_4 = QtWidgets.QFrame(self.frame_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_4.sizePolicy().hasHeightForWidth())
        self.frame_4.setSizePolicy(sizePolicy)
        self.frame_4.setStyleSheet("border:none;\n"
                                   "font: 12pt \"微软雅黑\";")
        self.frame_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_4.setObjectName("frame_4")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.frame_4)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.pushButton = QtWidgets.QPushButton(self.frame_4)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/icons/跌倒.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton.setIcon(icon)
        self.pushButton.setIconSize(QtCore.QSize(35, 35))
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout_2.addWidget(self.pushButton, 0, QtCore.Qt.AlignLeft)
        self.horizontalLayout.addWidget(self.frame_4)
        self.frame_5 = QtWidgets.QFrame(self.frame_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_5.sizePolicy().hasHeightForWidth())
        self.frame_5.setSizePolicy(sizePolicy)
        self.frame_5.setStyleSheet("QPushButton{\n"
                                   "    border:none;\n"
                                   "}\n"
                                   "QPushButton:hover{\n"
                                   "    padding-bottom:5px;\n"
                                   "}")
        self.frame_5.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_5.setObjectName("frame_5")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.frame_5)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.pushButton_2 = QtWidgets.QPushButton(self.frame_5)
        self.pushButton_2.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/icons/退出.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_2.setIcon(icon1)
        self.pushButton_2.setIconSize(QtCore.QSize(30, 30))
        self.pushButton_2.setObjectName("pushButton_2")
        self.horizontalLayout_3.addWidget(self.pushButton_2)
        self.pushButton_3 = QtWidgets.QPushButton(self.frame_5)
        self.pushButton_3.setText("")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/icons/缩小.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_3.setIcon(icon2)
        self.pushButton_3.setIconSize(QtCore.QSize(30, 30))
        self.pushButton_3.setObjectName("pushButton_3")
        self.horizontalLayout_3.addWidget(self.pushButton_3)
        self.pushButton_4 = QtWidgets.QPushButton(self.frame_5)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/icons/关闭.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_4.setIcon(icon3)
        self.pushButton_4.setIconSize(QtCore.QSize(30, 30))
        self.pushButton_4.setObjectName("pushButton_4")
        self.horizontalLayout_3.addWidget(self.pushButton_4)
        self.horizontalLayout.addWidget(self.frame_5, 0, QtCore.Qt.AlignRight)
        self.verticalLayout.addWidget(self.frame_2)
        self.frame_3 = QtWidgets.QFrame(self.frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(10)
        sizePolicy.setHeightForWidth(self.frame_3.sizePolicy().hasHeightForWidth())
        self.frame_3.setSizePolicy(sizePolicy)
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.frame_3)
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_4.setSpacing(0)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.frame_6 = QtWidgets.QFrame(self.frame_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(3)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_6.sizePolicy().hasHeightForWidth())
        self.frame_6.setSizePolicy(sizePolicy)
        self.frame_6.setMinimumSize(QtCore.QSize(230, 597))
        self.frame_6.setStyleSheet("#frame_6{\n"
                                   "    background-color: qlineargradient(spread:pad, x1:0.179, y1:0.982955, x2:1, y2:0, stop:0 rgba(153, 34, 57, 254), stop:1 rgba(255, 255, 255, 255));\n"
                                   "    border-bottom-left-radius:20px;\n"
                                   "}\n"
                                   "QPushButton{\n"
                                   "    border:none;\n"
                                   "    font-size:20px;        \n"
                                   "}\n"
                                   "QPushButton:pressed{\n"
                                   "    padding-top:4px;\n"
                                   "    padding-left:5px;\n"
                                   "\n"
                                   "}")
        self.frame_6.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_6.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_6.setObjectName("frame_6")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.frame_6)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.pushButton_Photo = QtWidgets.QPushButton(self.frame_6)
        self.pushButton_Photo.setStyleSheet("")
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap(":/icons/图片.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_Photo.setIcon(icon4)
        self.pushButton_Photo.setIconSize(QtCore.QSize(30, 30))
        self.pushButton_Photo.setObjectName("pushButton_Photo")
        self.verticalLayout_2.addWidget(self.pushButton_Photo)
        self.pushButton_Video = QtWidgets.QPushButton(self.frame_6)
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap(":/icons/视频.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_Video.setIcon(icon5)
        self.pushButton_Video.setIconSize(QtCore.QSize(30, 30))
        self.pushButton_Video.setObjectName("pushButton_Video")
        self.verticalLayout_2.addWidget(self.pushButton_Video)
        self.pushButton_Carera = QtWidgets.QPushButton(self.frame_6)
        icon6 = QtGui.QIcon()
        icon6.addPixmap(QtGui.QPixmap(":/icons/摄像机.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_Carera.setIcon(icon6)
        self.pushButton_Carera.setIconSize(QtCore.QSize(40, 40))
        self.pushButton_Carera.setObjectName("pushButton_Carera")
        self.verticalLayout_2.addWidget(self.pushButton_Carera)
        self.horizontalLayout_4.addWidget(self.frame_6)
        self.frame_7 = QtWidgets.QFrame(self.frame_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(10)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_7.sizePolicy().hasHeightForWidth())
        self.frame_7.setSizePolicy(sizePolicy)
        self.frame_7.setStyleSheet("")
        self.frame_7.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_7.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_7.setObjectName("frame_7")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.frame_7)
        self.horizontalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_5.setSpacing(0)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label = QtWidgets.QLabel(self.frame_7)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setMinimumSize(QtCore.QSize(765, 595))
        self.label.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.label.setText("")
        self.label.setObjectName("label")
        self.horizontalLayout_5.addWidget(self.label)
        self.horizontalLayout_4.addWidget(self.frame_7)
        self.verticalLayout.addWidget(self.frame_3)
        InterfaceWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(InterfaceWindow)
        self.pushButton_4.clicked.connect(InterfaceWindow.close)
        self.pushButton_3.clicked.connect(InterfaceWindow.showMinimized)
        QtCore.QMetaObject.connectSlotsByName(InterfaceWindow)

    def retranslateUi(self, InterfaceWindow):
        _translate = QtCore.QCoreApplication.translate
        InterfaceWindow.setWindowTitle(_translate("InterfaceWindow", "MainWindow"))
        self.pushButton.setText(_translate("InterfaceWindow", "Fall-Detection"))
        self.pushButton_Photo.setText(_translate("InterfaceWindow", "Photo"))
        self.pushButton_Video.setText(_translate("InterfaceWindow", "Video"))
        self.pushButton_Carera.setText(_translate("InterfaceWindow", "Carera"))

    def run(self, weights=ROOT / 'pretrained/best.pt',  # model.pt path(s)
            source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
            imgsz=640,  # inference size (pixels)
            conf_thres=0.25,  # confidence threshold
            iou_thres=0.45,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            device='0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            view_img=True,  # show results
            save_txt=False,  # save results to *.txt
            save_conf=False,  # save confidences in --save-txt labels
            save_crop=False,  # save cropped prediction boxes
            nosave=False,  # do not save images/videos
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            visualize=False,  # visualize features
            update=False,  # update all models
            project=ROOT / 'runs/detect',  # save results to project/name
            name='exp',  # save results to project/name
            exist_ok=False,  # existing project/name ok, do not increment
            line_thickness=3,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            half=False,  # use FP16 half-precision inference
            dnn=False,  # use OpenCV DNN for ONNX inference
            ):
        source = str(source)
        save_img = not nosave and not source.endswith('.txt')  # save inference images
        webcam = self.webcam

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        #
        # # Initialize

        # # Load model
        classify = False
        imgsz = check_img_size(imgsz, s=self.stride)  # check image size

        # Dataloader
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=self.stride, auto=True)
            bs = len(dataset)  # batch_size
            self.cap = dataset.cap
            self.camflag = True
            self.pushButton_Carera.setText("Stop")
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=self.stride, auto=True)
            bs = 1  # batch_size
        vid_path, vid_writer = [None] * bs, [None] * bs

        dt, seen = [0.0, 0.0, 0.0], 0
        for path, img, im0s, vid_cap in dataset:
            t1 = time_sync()
            if self.onnx:
                img = img.astype('float32')
            else:
                img = torch.from_numpy(img).to(self.device)
                img = img.half() if half else img.float()  # uint8 to fp16/32
            img = img / 255.0  # 0 - 255 to 0.0 - 1.0
            if len(img.shape) == 3:
                img = img[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference

            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = self.model(img, augment=augment, visualize=visualize)[0]
            t3 = time_sync()
            dt[1] += t3 - t2

            # NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            dt[2] += time_sync() - t3

            # Second-stage classifier (optional)
            if classify:
                pred = apply_classifier(pred, self.modelc, img, im0s)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                if webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
                else:
                    p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # img.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + (
                    '' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                names = self.names
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                        # Check if class is "Fall" and print message
                        if names[int(cls)] == 'Fall':
                            print("Fall detected!")  # Print message when "Fall" is detected

                # Print time (inference-only)
                # print(f'{s}Done. ({t3 - t2:.3f}s)')

                # Stream results
                self.im0 = annotator.result()
                if view_img:
                    self.result = cv2.cvtColor(self.im0, cv2.COLOR_BGR2BGRA)
                    # self.result = cv2.resize(self.result, (640, 480), interpolation=cv2.INTER_AREA)
                    self.QtImg = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0],
                                              QtGui.QImage.Format_RGB32)
                    self.label.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))
                    self.label.setScaledContents(True)

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, self.im0)
                    else:  # 'video' or 'stream'
                        if vid_path[i] != save_path:  # new video
                            vid_path[i] = save_path
                            if isinstance(vid_writer[i], cv2.VideoWriter):
                                vid_writer[i].release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, self.im0.shape[1], self.im0.shape[0]
                                save_path += '.mp4'
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer[i].write(self.im0)

            if self.stopEvent.is_set() == True:
                if self.cap is not None and self.cap.isOpened():
                    self.cap.release()
                self.stopEvent.clear()
                self.initLogo()
                break
        # Print results
        t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
        # print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            print(f"Results saved to {colorstr('bold', save_dir)}{s}")
        if update:
            strip_optimizer(weights)  # update model (to fix SourceChangeWarning)
        if source.split(".")[1] == "mp4":
            self.update_button_states(True)
            self.initLogo()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ui = Ui_MainWindow()
    sys.exit(app.exec_())
