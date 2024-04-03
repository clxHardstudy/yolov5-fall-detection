import os
import cv2
import sys
import time
import torch
import base64
import requests
import threading
from newui import res_rc
from pathlib import Path
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt
from threading import Thread
import torch.backends.cudnn as cudnn
from PyQt5 import QtCore, QtGui, QtWidgets
from Ui_Window import Ui_FallDetectWindow, Ui_LoginWindow
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QMessageBox

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
class Ui_MainWindow(Ui_FallDetectWindow):
    def __init__(self, parent=None):
        super().__init__()

        self.fall_detected_time = None  # 记录跌倒检测时间
        self.five_seconds_passed = False  # 标记是否超过了5秒
        self.uploaded = False  # 标记是否已经上传图片

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

    # photo按键点击发生事件
    def button_photo_open(self):
        print('button_photo_open')
        img_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择图片", "",
                                                            "Image Files (*.png *.jpg *.bmp *.jpeg *.gif *.tif)")
        if not img_name:
            return
        thread1 = Thread(target=self.run, kwargs={"weights": self.weights, "source": str(img_name), "nosave": True,
                                                  "view_img": True})
        thread1.start()

    # video按键点击发生事件
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

    # camera按键点击发生事件
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

    def upload_image_to_minio(self, base64_img):
        url = "http://127.0.0.1:8000/file/MinioFileBytes"
        payload = {
            "filebase64str": str(base64_img)
        }
        response = requests.post(url=url, json=payload)
        if response.status_code == 200:
            print("Image uploaded to Minio")
        else:
            print("Failed to upload image to Minio")

    def detect_and_upload(self, im0):
        _, img_buffer = cv2.imencode('.jpg', im0)
        base64_img = base64.b64encode(img_buffer).decode('utf-8')
        upload_thread = threading.Thread(target=self.upload_image_to_minio, args=(base64_img,))
        upload_thread.start()

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
                            if self.fall_detected_time is None:
                                # 给跌倒时刻赋一个初始值
                                self.fall_detected_time = time.time()
                            else:
                                current_time = time.time()
                                elapsed_time = current_time - self.fall_detected_time
                                if elapsed_time >= 5:  # 如果时间超过5秒
                                    # 上传图片到minio
                                    self.detect_and_upload(im0)
                                    self.fall_detected_time = None
                        # 必须是持续的5s才可以上传一次图片：这个else防止了第一次fall之后很长时间之后又fall了一次的情况
                        else:
                            self.fall_detected_time = None
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


class LoginWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_LoginWindow()
        self.ui.setupUi(self)
        # 隐藏多余的窗体
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        # 添加阴影
        self.shadow = QtWidgets.QGraphicsDropShadowEffect(self)
        self.shadow.setOffset(0, 0)
        self.shadow.setBlurRadius(15)
        self.shadow.setColor(Qt.black)
        self.ui.frame.setGraphicsEffect(self.shadow)
        # 登录点击展示窗口
        self.ui.pushButton_Login.clicked.connect(lambda: self.ui.stackedWidget_2.setCurrentIndex(1))
        # 注册点击展示窗口
        self.ui.pushButton_Register.clicked.connect(lambda: self.ui.stackedWidget_2.setCurrentIndex(0))
        self.ui.pushButton_L_Sure.clicked.connect(self.login_in)
        self.ui.pushButton_R_Sure.clicked.connect(self.register)
        self.show()

    def login_in(self):
        username = self.ui.lineEdit_L_Account.text().replace(" ", "")
        password = self.ui.lineEdit_L_Password.text().replace(" ", "")
        if username == "" or password == "":
            QMessageBox.warning(self, "登录失败", "账号或密码是必要的！", QMessageBox.Ok)
            return
        url = "http://127.0.0.1:8000/user/login"
        payload = {
            "username": username,
            "password": password,
        }
        response = requests.post(url=url, json=payload)
        if response.json()["status_code"] == 200:
            self.win = Ui_MainWindow()
            self.close()
        elif response.json()["status_code"] == 404:
            QMessageBox.warning(self, "登录失败", "用户不存在！", QMessageBox.Ok)
        elif response.json()["status_code"] == 401:
            QMessageBox.warning(self, "登录失败", "账号或密码错误！", QMessageBox.Ok)
        else:
            QMessageBox.warning(self, "登录失败", "后端接口服务异常，请检查后端程序！", QMessageBox.Ok)

    def register(self):
        username = self.ui.lineEdit_R_Account.text().replace(" ", "")
        password1 = self.ui.lineEdit_R_Password1.text().replace(" ", "")
        password2 = self.ui.lineEdit_R_Password2.text().replace(" ", "")
        if username == "" or password1 == "" or password2 == "":
            QMessageBox.warning(self, "注册失败", "账号或密码是必要的！", QMessageBox.Ok)
            return
        if password1 != password2:
            QMessageBox.warning(self, "注册失败", "两次输入的密码不一致！！", QMessageBox.Ok)

        url = "http://127.0.0.1:8000/user"
        payload = {
            "username": username,
            "password": password1,
        }
        response = requests.post(url=url, json=payload)
        print(response.json())
        if response.json()["exist"] == "True":
            QMessageBox.warning(self, "注册失败", "该用户已经存在！", QMessageBox.Ok)
            return
        else:
            QMessageBox.warning(self, "注册成功", "请准备登录！", QMessageBox.Ok)
            self.ui.lineEdit_R_Account.clear()
            self.ui.lineEdit_R_Password1.clear()
            self.ui.lineEdit_R_Password2.clear()
            self.ui.pushButton_Login.click()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ui = LoginWindow()
    sys.exit(app.exec_())
