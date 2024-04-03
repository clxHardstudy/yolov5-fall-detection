from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import *
import os
import sys
from pathlib import Path
import threading
from threading import Thread

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
        self.setupUi(self)
        self.setWindowIcon(QIcon("UI/icon.png"))
        self.initLogo()
        self.initSlots()

        self.device = '0'
        self.weights = "pretrained/best.pt"
        self.source = " "
        self.imgsz = 640
        self.half = False

        self.names = None
        self.webcam = False
        self.camflag = False
        self.stopEvent = threading.Event()
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
        self.initWeight()

    def initSlots(self):
        self.picButton.clicked.connect(self.button_image_open)
        self.camButton.clicked.connect(self.button_camera_open)
        self.weightButton.clicked.connect(self.button_weight_open)

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

    def initLogo(self):
        pix = QtGui.QPixmap('UI/YOLO.png')
        self.label.setScaledContents(True)
        self.label.setPixmap(pix)
        print("initLogo")

    def button_image_open(self):
        print('button_image_open')
        if self.picButton.text() == "Stop":
            self.stopEvent.set()
            self.picButton.setText("Photo\Video")
            self.camButton.setEnabled(True)
            self.weightButton.setEnabled(True)

        else:
            img_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择图片或视频", "", "")
            if not img_name:
                return
            if img_name.endswith("mp4"):
                self.picButton.setText("Stop")
                self.camButton.setEnabled(False)
                self.weightButton.setEnabled(False)
            thread1 = Thread(target=self.run, kwargs={"weights": self.weights, "source": str(img_name), "nosave": True,
                                                      "view_img": True})

            thread1.start()

    def button_camera_open(self):
        if self.camflag == False:
            print('button_camera_open')
            self.webcam = True
            self.picButton.setEnabled(False)
            self.weightButton.setEnabled(False)
            thread2 = Thread(target=self.run,
                             kwargs={"weights": self.weights, "source": "0", "nosave": True, "view_img": True})
            thread2.start()
        else:
            print('button_camera_close')
            self.stopEvent.set()
            self.camflag = False
            self.webcam = False
            self.picButton.setEnabled(True)
            self.weightButton.setEnabled(True)
            self.camButton.setText("Camera")
            self.initLogo()

    def button_weight_open(self):
        print('button_weight_open')
        weight_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择权重", "", "*.pt")  # All Files(*)
        if not weight_name:
            return
        self.weights = str(weight_name)
        self.initWeight()

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        MainWindow.setFixedSize(900, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setContentsMargins(20, -1, 20, -1)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.picButton = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.picButton.sizePolicy().hasHeightForWidth())
        self.picButton.setSizePolicy(sizePolicy)
        self.picButton.setMinimumSize(QtCore.QSize(150, 100))
        self.picButton.setMaximumSize(QtCore.QSize(150, 100))
        self.picButton.setSizeIncrement(QtCore.QSize(0, 0))
        self.picButton.setBaseSize(QtCore.QSize(0, 0))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(21)
        self.picButton.setFont(font)
        self.picButton.setObjectName("picButton")
        self.verticalLayout.addWidget(self.picButton)
        self.camButton = QtWidgets.QPushButton(self.centralwidget)
        self.camButton.setMinimumSize(QtCore.QSize(150, 100))
        self.camButton.setMaximumSize(QtCore.QSize(150, 100))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(21)
        self.camButton.setFont(font)
        self.camButton.setObjectName("pushButton")
        self.verticalLayout.addWidget(self.camButton)
        self.weightButton = QtWidgets.QPushButton(self.centralwidget)
        self.weightButton.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.weightButton.sizePolicy().hasHeightForWidth())
        self.weightButton.setSizePolicy(sizePolicy)
        self.weightButton.setMinimumSize(QtCore.QSize(150, 100))
        self.weightButton.setMaximumSize(QtCore.QSize(150, 100))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(21)
        self.weightButton.setFont(font)
        self.weightButton.setObjectName("weightButton")
        self.verticalLayout.addWidget(self.weightButton)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setText("")
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.horizontalLayout.setStretch(0, 1)
        self.horizontalLayout.setStretch(1, 3)
        self.horizontalLayout_2.addLayout(self.horizontalLayout)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 828, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "目标检测Demo"))
        self.picButton.setText(_translate("MainWindow", "Photo\Video"))
        self.camButton.setText(_translate("MainWindow", "Camera"))
        self.weightButton.setText(_translate("MainWindow", "Weights"))

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
            self.camButton.setText("Stop")
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


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ui = Ui_MainWindow()
    ui.show()
    sys.exit(app.exec_())
