import argparse
import msvcrt
import os
import os.path as osp
import time
import cv2
import torch
import math
import numpy as np
import pandas as pd
from  collections import deque


from loguru import logger
from pynput import keyboard

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer
# from GSI import GSInterpolation
import tracklet
import sys
# from PyQt5.QtWidgets import QApplication, QMainWindow

# import hello
import demo_track


def click_success():
    print("i'm your father")
def cl():
    print("i'm your mom")


IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
# global np_store
# np_store=np.zeros((5000,6))






def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    parser.add_argument(
        "--demo", default="video", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        #"--path", default="./datasets/mot/train/MOT17-05-FRCNN/img1", help="path to images or video"
        "--path", default="sao_2.mp4", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        default=True,
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default="exps/example/mot/yolox_s_mix_det.py",
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("-c", "--ckpt", default="sao.pth", type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.5, type=float, help="test conf")
    parser.add_argument("--nms", default=0.55, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=2048, type=int, help="test img size")
    parser.add_argument("--fps", default=8, type=int, help="frame rate (fps)")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=True,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=True,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.6, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=8, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.99, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=3,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        trt_file=None,
        decoder=None,
        device=torch.device("cpu"),
        fp16=False
    ):
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones((1, 3, exp.test_size[0], exp.test_size[1]), device=device)
            self.model(x)
            self.model = model_trt
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        # t1 = time.time()
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img
        # mask = np.zeros(img.shape[:2], dtype=np.uint8)
        #
        # # 定义圆形区域的中心坐标和半径
        # center = (img.shape[1] // 2, img.shape[0] // 2)
        # radius = min(center[0], center[1])
        #
        # # 在掩码上绘制圆形
        # cv2.circle(mask, center, radius, (255, 255, 255), -1)
        #
        # # 将掩码应用于输入图像
        # masked_img = cv2.bitwise_and(img, img, mask=mask)
        # t1 = time.time()
        img, ratio = preproc(img, self.test_size)


        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # to FP16
        # logger.info("Infer time: {:.4f}s".format(time.time() - t1))
        with torch.no_grad():

            t0=timer.tic()
            outputs = self.model(img)

            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )

            # t2=time.time()
            # logger.info("Infer time: {:.4f}s".format(time.time() - t1))
        return outputs, img_info

    # def inference_(self, img, timer):
    #
    #     img_info = {"id": 0}
    #     if isinstance(img, str):
    #         img_info["file_name"] = osp.basename(img)
    #         img = cv2.imread(img)
    #     else:
    #         img_info["file_name"] = None
    #
    #     height, width = img.shape[:2]
    #     img_info["height"] = height
    #     img_info["width"] = width
    #     img_info["raw_img"] = img
    #
    #     img, ratio = preproc(img, self.test_size)
    #     t0 = timer.tic()
    #     img_info["ratio"] = ratio
    #     img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
    #
    #     if self.fp16:
    #         img = img.half()  # to FP16
    #
    #     with torch.no_grad():
    #         # t0=timer.tic()
    #         outputs = self.model(img)
    #         if self.decoder is not None:
    #             outputs = self.decoder(outputs, dtype=outputs.type())
    #         outputs = postprocess(
    #             outputs, self.num_classes, self.confthre, self.nmsthre
    #         )
    #         # logger.info("Infer time: {:.4f}s".format(timer.toc() - t0))
    #     return outputs, img_info


def image_demo(predictor, vis_folder, current_time, args):
    if osp.isdir(args.path):
        files = get_image_list(args.path)
    else:
        files = [args.path]
    files.sort()
    tracker = BYTETracker(args, frame_rate=args.fps)
    timer = Timer()
    results = []

    for frame_id, img_path in enumerate(files, 1):
        outputs, img_info = predictor.inference(img_path, timer)
        if outputs[0] is not None:
            online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                # vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                # if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                online_scores.append(t.score)
                    # save results
                results.append(
                        f"{frame_id+1},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},1,-1\n"
                    )
            timer.toc()
            online_im = plot_tracking(
                img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id, fps=1. / timer.average_time
            )
        else:
            timer.toc()
            online_im = img_info['raw_img']

        # result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        if args.save_result:
            timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            save_folder = osp.join(vis_folder, timestamp)
            os.makedirs(save_folder, exist_ok=True)
            cv2.imwrite(osp.join(save_folder, osp.basename(img_path)), online_im)

        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break

    if args.save_result:
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")


def imageflow_demo(predictor, vis_folder, current_time, args):
    # COLORS_10 = [(144, 238, 144), (178, 34, 34), (221, 160, 221), (0, 255, 0), (0, 128, 0), (210, 105, 30),
    #              (220, 20, 60),
    #              (192, 192, 192), (255, 228, 196), (50, 205, 50), (139, 0, 139), (100, 149, 237), (138, 43, 226),
    #              (238, 130, 238),
    #              (255, 0, 255), (0, 100, 0), (127, 255, 0), (255, 0, 255), (0, 0, 205), (255, 140, 0), (255, 239, 213),
    #              (199, 21, 133), (124, 252, 0), (147, 112, 219), (106, 90, 205), (176, 196, 222), (65, 105, 225),
    #              (173, 255, 47),
    #              (255, 20, 147), (219, 112, 147), (186, 85, 211), (199, 21, 133), (148, 0, 211), (255, 99, 71),
    #              (144, 238, 144),
    #              (255, 255, 0), (230, 230, 250), (0, 0, 255), (128, 128, 0), (189, 183, 107), (255, 255, 224),
    #              (128, 128, 128),
    #              (105, 105, 105), (64, 224, 208), (205, 133, 63), (0, 128, 128), (72, 209, 204), (139, 69, 19),
    #              (255, 245, 238),
    #              (250, 240, 230), (152, 251, 152), (0, 255, 255), (135, 206, 235), (0, 191, 255), (176, 224, 230),
    #              (0, 250, 154),
    #              (245, 255, 250), (240, 230, 140), (245, 222, 179), (0, 139, 139), (143, 188, 143), (255, 0, 0),
    #              (240, 128, 128),
    #              (102, 205, 170), (60, 179, 113), (46, 139, 87), (165, 42, 42), (178, 34, 34), (175, 238, 238),
    #              (255, 248, 220),
    #              (218, 165, 32), (255, 250, 240), (253, 245, 230), (244, 164, 96), (210, 105, 30)]
    # global np_store
    # global summer
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    # print(fps)
    # print((width,height))
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    save_folder = osp.join(vis_folder, timestamp)
    os.makedirs(save_folder, exist_ok=True)

    if args.demo == "video":
        save_path = osp.join(save_folder, args.path.split("/")[-1])
    else:
        save_path = osp.join(save_folder, "camera.mp4")
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    tracker = BYTETracker(args, frame_rate=30)
    timer = Timer()
    frame_id = 0
    results = []
    dict_box=dict()

    while True:
        """some code"""
        # print(frame_id)
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        ret_val, frame = cap.read()
        # 定义圆形区域的中心坐标和半径
        # center = (frame.shape[1] // 2, frame.shape[0] // 2)
        # radius = min(center[0], center[1])
        # cv2.circle(frame,center,radius,(0,0,255))

        # print(frame.shape)
        if ret_val:


            outputs, img_info  = predictor.inference(frame, timer)


            if outputs[0] is not None:

                online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)


                online_tlwhs = []

                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    # vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    # if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)

                    results.append(
                            f"{frame_id+1},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},1,-1\n"
                        )


                online_im = plot_tracking(
                    img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1, fps=0
                )
                timer.toc()
                # print(online_im)
                # print(img_info['raw_img'].shape)

                # print((img_info['raw_img']))
                # print(online_ids)
                # for j in range(len(online_tlwhs)):
                #     x_center = online_tlwhs[j][0] + online_tlwhs[j][2] / 2
                #     y_center = online_tlwhs[j][1] + online_tlwhs[j][3] / 2
                #     id = online_ids[j]
                #     center = [x_center, y_center]
                #     dict_box.setdefault(id, []).append(center)
                # for key,value in dict_box.items():
                #     if len(value)>50:
                #         del value[0]
                #
                #
                #
                # # print(frame_id)
                #     # print((dict_box))
                # if frame_id > 0:
                #     summer = 0
                #     speed =0
                #
                #     for key,value in dict_box.items():
                #             # np_store[key][0]=key
                #             for a in range(len(value) - 1):
                #                 color = COLORS_10[key % len(COLORS_10)]
                #
                #
                #                 # if len(value)==np_store[key][3]:
                #                 #     # print(key,len(value))
                #                 #     continue
                #                 # else:
                #                 index_start = a
                #                  # first_loc_x=value[index_start][0]
                #                  # first_loc_y=value[index_start][1]
                #                 index_end = index_start +1
                #
                #
                #                 cv2.line(online_im, tuple(map(int, value[index_start])), tuple(map(int, value[index_end])),
                #                              # map(int,"1234")转换为list[1,2,3,4]
                #                             color, thickness=2, lineType=8)
                #
                #
                #                     second_loc_x=value[index_end][0]
                #                     second_loc_y=value[index_end][1]
                #                     summer= np_store[key][1]+math.sqrt(((first_loc_x-second_loc_x)**2)+((first_loc_y-second_loc_y)**2))
                #                     speed =summer/(len(value)*(1/fps))
                #                     np_store[key][1]=round(summer,2)
                #                     np_store[key][2]=round(speed,2)
                #                     np_store[key][3]=len(value)
                #                     summer =0







            else:
                timer.toc()
                online_im = img_info['raw_img']
            if args.save_result:
                vid_writer.write(online_im)

            cv2.namedWindow("img", 0);
            cv2.resizeWindow("img", 4096, 4096);
            cv2.imshow("img", online_im)
            if cv2.waitKey(1) ==27:
                break
        else:
            break

        if msvcrt.kbhit():
            if ord(msvcrt.getch()) != None:
                break


        frame_id += 1
    # pd_data=pd.DataFrame(np_store,columns=['key','summ','speed','life','4','5'])
    # pd_data.to_csv(f"{timestamp}.csv")

    if args.save_result:
        res_file = osp.join(vis_folder, f"{timestamp}.txt")

        with open(res_file, 'w') as f:
            f.writelines(results)
        # GSInterpolation(path_in=res_file,path_out=f"track/{timestamp}_gsl.csv",path_out_=f"track/{timestamp}_gsl_md.csv",interval=8,tau=1,fps=fps)
        # tracklet.huatu(f"track/{timestamp}_gsl.csv",f"D:/GSL/{timestamp}_gsl.png")
        # logger.info(f"save results to track/{timestamp}_gsl.csv and track/{timestamp}_gsl_md.csv")


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    output_dir = osp.join(exp.output_dir, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    if args.save_result:
        vis_folder = osp.join(output_dir, "track_vis")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"
    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model().to(args.device)
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = osp.join(output_dir, "best_ckpt.pth.tar")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.fp16:
        model = model.half()  # to FP16

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = osp.join(output_dir, "model_trt.pth")
        assert osp.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(model, exp, trt_file, decoder, args.device, args.fp16)
    current_time = time.localtime()
    if args.demo == "image":
        image_demo(predictor, vis_folder, current_time, args)
    elif args.demo == "video" or args.demo == "webcam":
        imageflow_demo(predictor, vis_folder, current_time, args)


if __name__ == "__main__":



    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    main(exp, args)

