import os
import sys
import cv2
import pickle
import torch
import argparse
from glob import  glob
from tqdm import tqdm
import numpy as np
import pandas as pd
from time import time
import albumentations as A

from model.resnet import ResNet
from model.mstcn import MultiStageModel
from model.transformer import Transformer
from utils.parser import ParserUse

from torch.utils.data import DataLoader
from dataset.esd import VideoSample

phase_dict = {}
phase_dict_key = ['idle', 'marking', 'injection', 'dissection']
for i in range(len(phase_dict_key)):
    phase_dict[phase_dict_key[i]] = i

label_dict = {}
phase_dict_key = ['idle', 'marking', 'injection', 'dissection']
for i, phase in enumerate(phase_dict_key):
    label_dict[i] = phase

class PhaseSeg(object):
    """
    The class performs generic object detection on a video file.
    It uses yolo5 pretrained model to make inferences and opencv2 to manage frames.
    Included Features:
    1. Reading and writing of video file using  Opencv2
    2. Using pretrained model to make inferences on frames.
    3. Use the inferences to plot boxes on objects along with labels.
    Upcoming Features:
    """
    def __init__(self, input_file, hypers, out_file=None, label_file=None, quiet=False, arg=None):
        """
        :param input_file: provide youtube url which will act as input for the model.
        :param out_file: name of a existing file, or a new file in which to write the output.
        :return: void
        """
        self.input_file = input_file
        self.hypers = hypers
        self.arg = arg
        self.model = self.load_model()
        self.out_file = os.path.join("./results", os.path.basename(input_file)) if out_file is None else out_file
        self.frame_feature_cache = None
        self.frame_cache_len = 2 ** (self.arg.mstcn_layers + 1) - 1
        self.temporal_feature_cache = None
        self.label2phase_dict = label_dict
        self.aug = A.Compose([
            A.Resize(250, 250),
            A.CenterCrop(224, 224),
            A.Normalize()
        ])
        self.label_file = label_file
        self.quiet = quiet
        if label_file is not None:
            self.labels = self.get_labels()
        else:
            self.labels = None

    def get_video_from_file(self):
        """
        Function creates a streaming object to read the video from the file frame by frame.
        :param self:  class object
        :return:  OpenCV object to stream video frame by frame.
        """
        cap = cv2.VideoCapture(self.input_file)
        assert cap is not None
        return cap

    def get_labels(self):
        assert self.label_file, "Label file {} does not exit".format(self.label_file)
        phase_label = pd.read_csv(self.label_file, header=None, sep="[ ]{1,}|\t", engine="python")
        if len(phase_label.columns) == 5:
            phase_label.columns = ["Frame", "Phase", "#1", "#2", "#3"]
        elif len(phase_label.columns) == 2:
            phase_label.columns = ["Frame", "Phase"]
        else:
            raise ValueError("The header of label file cannot be matched")
        phase_label = phase_label.astype({"Frame": int, "Phase": str})
        phase_label = phase_label.replace({"Phase": phase_dict})
        phase_labels = phase_label["Phase"].tolist()
        return phase_labels

    def save_preds(self, preds):

        pd_label = pd.DataFrame({"Frame": list(range(1, len(preds) + 1, 1)), "Phase": preds})
        pd_label = pd_label.astype({"Frame": "int", "Phase": "str"})
        save_file = self.out_file.replace(".avi", ".txt")
        print(save_file)
        pd_label.to_csv(save_file, index=False, header=None, sep="\t")

    def load_model(self):
        """
        Function loads the yolo5 model from PyTorch Hub.
        """
        self.resnet = ResNet(out_channels=self.hypers.out_classes, has_fc=False)
        paras = torch.load(self.hypers.resnet_model)["model"]
        paras = {k: v for k, v in paras.items() if "fc" not in k}
        paras = {k: v for k, v in paras.items() if "embed" not in k}
        self.resnet.load_state_dict(paras, strict=True)
        self.resnet.cuda()
        self.resnet.eval()

        self.fusion = MultiStageModel(mstcn_stages=self.hypers.mstcn_stages, mstcn_layers=self.hypers.mstcn_layers,
                                      mstcn_f_maps=self.hypers.mstcn_f_maps, mstcn_f_dim=self.hypers.mstcn_f_dim,
                                      out_features=self.hypers.out_classes, mstcn_causal_conv=True, is_train=False)
        paras = torch.load(self.hypers.fusion_model)
        self.fusion.load_state_dict(paras)
        self.fusion.cuda()
        self.fusion.eval()

        self.transformer = Transformer(self.hypers.mstcn_f_maps, self.hypers.mstcn_f_dim, self.hypers.out_classes, self.hypers.trans_seq, d_model=self.hypers.mstcn_f_maps)
        paras = torch.load(self.hypers.trans_model)
        self.transformer.load_state_dict(paras)
        self.transformer.cuda()
        self.transformer.eval()

    def cache_frame_features(self, feature):
        if self.frame_feature_cache is None:
            self.frame_feature_cache = feature
        elif self.frame_feature_cache.shape[0] > self.frame_cache_len:
            self.frame_feature_cache = torch.cat([self.frame_feature_cache[1:], feature], dim=0)
        else:
            self.frame_feature_cache = torch.cat([self.frame_feature_cache, feature], dim=0)
        return self.frame_feature_cache

    def seg_frame(self, frame):
        """
        function scores each frame of the video and returns results.
        :param frame: frame to be infered.
        :return: labels and coordinates of objects found.
        """
        frame = self.aug(image=frame)["image"]
        with torch.no_grad():
            frame = np.expand_dims(np.transpose(frame, [2, 0, 1]), axis=0)
            frame = torch.tensor(frame).cuda()
            frame_feature = self.resnet(frame)
            # print(frame_feature.size())
            cat_frame_feature = self.cache_frame_features(frame_feature).unsqueeze(0)
            temporal_feature = self.fusion(cat_frame_feature.transpose(1, 2))

            # Temporal feature: [1, 5, 512], Frame feature：[1, 512, 2048]
            pred = self.transformer(temporal_feature.detach(), cat_frame_feature)[-1].cpu().numpy()
        return self.label2phase_dict[np.argmax(pred, axis=0)]

    def plot_boxes(self, results, frame):
        """
        plots boxes and labels on frame.
        :param results: inferences made by model
        :param frame: frame on which to  make the plots
        :return: new frame with boxes and labels plotted.
        """
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
            bgr = (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 1)
            label = f"{int(row[4]*100)}"
            cv2.putText(frame, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.putText(frame, f"Total Targets: {n}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.1, (0, 255, 0), 1)

        return frame

    def add_text(self, fc, results, fps, frame):
        cv2.putText(frame, "Frame:{:>6d}".format(fc), (30, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, "Phase:{:>15s}".format(results), (30, 60),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, "FPS:{:>8.2f}".format(fps), (30, 90),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return frame

    def __call__(self):
        player = self.get_video_from_file() # create streaming service for application
        assert player.isOpened()
        x_shape = int(player.get(cv2.CAP_PROP_FRAME_WIDTH))
        y_shape = int(player.get(cv2.CAP_PROP_FRAME_HEIGHT))
        four_cc = cv2.VideoWriter_fourcc(*"XVID")

        out = cv2.VideoWriter(self.out_file, four_cc, 8, (x_shape, y_shape), True)

        tfcc = 0
        preds = []
        while True:
            tfcc += 1
            start_time = time()
            ret, frame = player.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.seg_frame(frame)
            preds.append(results)
            end_time = time()
            if tfcc % 10 == 1:
                fps = 1/np.round(end_time - start_time, 3)
                print("{:10.5f}".format(fps))
            frame = self.add_text(tfcc, results, fps, frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame)
            if not self.quiet:
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        player.release()
        self.save_preds(preds)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("-f", required=False, default=None, type=str, help="Target video to be processed")
    parse.add_argument("-d", default=None, type=str, help="File to save processed video")
    parse.add_argument("-q", default=False, action='store_true', help="Display video")
    parse.add_argument("--cfg", default="train", type=str)

    cfg = parse.parse_args()
    cfg = ParserUse(cfg.cfg, "stream").add_args(cfg)

    if cfg.f is None:
        videos = sorted(glob("/research/dept8/rshr/jfcao/Dataset/ESD_new_data/mini_avi/*.avi"))
        for case_idx in cfg.test_names:
            print(videos[case_idx])
            cfg.f = videos[case_idx]
            phase_seg = PhaseSeg(cfg.f, cfg, cfg.d, quiet=cfg.q, arg=cfg)
            phase_seg()
    else:
        phase_seg = PhaseSeg(cfg.f, cfg, cfg.d, quiet=cfg.q, arg=cfg)


    #
    # with open(cfg.data_file, "rb") as f:
    #     data_dict = pickle.load(f)
    # with open(cfg.emb_file, "rb") as f:
    #     emb_dict = pickle.load(f)
    #
    # test_data = VideoSample(data_dict=data_dict, data_idxs=cfg.test_names, data_features=emb_dict, is_train=False, get_name=True)
    # test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
    #
    # import os
    # import pandas as pd
    # from sklearn import metrics
    # with torch.no_grad():
    #     for data in tqdm(test_loader, desc="Predicting"):
    #         img_featrues0, img_names = data[0].cuda(non_blocking=True), data[1]
    #         img_featrues = torch.transpose(img_featrues0, 1, 2)
    #         features = phase_seg.fusion(img_featrues)[-1].squeeze(1)  # Shifted predictions for all frames
    #         p_classes = phase_seg.transformer(features.detach(), img_featrues0).squeeze()
    #         preds = torch.argmax(p_classes, dim=-1).cpu().numpy().tolist()
    #
    #         gt_label_file = os.path.join(cfg.label_dir, os.path.basename(img_names[0][0].split("-")[0]) + ".txt")
    #         gt_label = pd.read_csv(gt_label_file, header=None, sep="\t", names=["Frame", "Phase"], index_col=False)
    #         gt_label = gt_label.replace({"Phase": phase_dict})
    #         gt_label = gt_label["Phase"].tolist()
    #         acc = metrics.accuracy_score(gt_label, preds)
    #
    #         print(set(preds))
    #         print("Accuracy: ", acc)

            # break