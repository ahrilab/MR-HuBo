import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from openpifpaf import decoder as ppdecoder
from openpifpaf import network as ppnetwork
from openpifpaf.predictor import Predictor
from openpifpaf.stream import Stream

from pymaf_x.models import pymaf_net
from pymaf_x.core import path_config
from pymaf_x.datasets.inference import Inference

PYMAF_MODEL_WEIGHT_PATH = "./data/pymaf_x/PyMAF-X_model_checkpoint_v1.1.pt"


def load_pymaf_model(weight_path):
    """
    Load PyMAF model with pretrained weights.

    return:
        model: PyMAF model with pretrained weights
    """

    # set device info
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("using {} as a device...".format(device))

    # ========= Define model ========= #
    # TODO: SMPL_MEAN_PARAMS 관련 에러 핸들링하기
    pymaf_model = pymaf_net(path_config.SMPL_MEAN_PARAMS, is_train=False).to(device)

    # ========= Load pretrained weights ========= #
    checkpoint = torch.load(weight_path)
    pymaf_model.load_state_dict(checkpoint["model"], strict=True)
    print(f"loaded checkpoint: {weight_path}")

    pymaf_model.eval()

    return pymaf_model


def get_pp_args(device):
    """
    Get arguments for openpifpaf.
    """
    pp_parser = argparse.ArgumentParser()
    ppnetwork.Factory.cli(pp_parser)
    ppdecoder.cli(pp_parser)
    Predictor.cli(pp_parser)
    Stream.cli(pp_parser)
    pp_args = pp_parser.parse_args(args=[])
    pp_args.detector_checkpoint = 'shufflenetv2k30-wholebody'
    pp_args.device = device
    pp_args.detector_batch_size = 1
    pp_args.detection_threshold = 0.55
    pp_args.force_complete_pose = True

    return pp_args


def infer_smpl_with_pymaf(images):
    """
    Infer SMPL parameters from a RGB image with PyMAF-X.

    Args:
        images: input images
        args: arguments

    return:
        smpl_params: SMPL pose parameters
    """
    
    # Copy the PIL Image into a numpy array
    images_array = [np.array(img, dtype=np.float32) for img in images]

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # args.device = device

    # Load PyMAF-X model
    pymaf_model = load_pymaf_model(PYMAF_MODEL_WEIGHT_PATH)

    # Person Detection with OpenPifPaf
    pp_args = get_pp_args(device)
    ppdecoder.configure(pp_args)
    ppnetwork.Factory.configure(pp_args)
    ppnetwork.Factory.checkpoint = pp_args.detector_checkpoint
    Predictor.configure(pp_args)
    Stream.configure(pp_args)

    Predictor.batch_size = pp_args.detector_batch_size
    if pp_args.detector_batch_size > 1:
        Predictor.long_edge = 1000
    Predictor.loader_workers = 1
    predictor = Predictor()

    capture = predictor.pil_images(images)

    tracking_results = {}
    num_frames = len(images)
    print("Running openpifpaf for person detection...")
    for preds, _, meta in tqdm(capture, total=num_frames // pp_args.detector_batch_size):

        for pid, ann in enumerate(preds):
            if ann.score > pp_args.detection_threshold:
                frame_i = (
                    meta["frame_i"] - 1 if "frame_i" in meta else meta["dataset_index"]
                )
                person_id = f"f_{frame_i}_p_{pid}"
                det_wb_kps = ann.data
                det_face_kps = det_wb_kps[23:91]
                tracking_results[person_id] = {
                    "frames": [frame_i],
                    # 'predictions': [ann.json_data() for ann in preds]
                    "joints2d": [det_wb_kps[:17]],
                    "joints2d_lhand": [det_wb_kps[91:112]],
                    "joints2d_rhand": [det_wb_kps[112:133]],
                    "joints2d_face": [
                        np.concatenate([det_face_kps[17:], det_face_kps[:17]])
                    ],
                    "vis_face": [np.mean(det_face_kps[17:, -1])],
                    "vis_lhand": [np.mean(det_wb_kps[91:112, -1])],
                    "vis_rhand": [np.mean(det_wb_kps[112:133, -1])],
                }

    """
    PifPaf Results:
      - joints2d: 17 keypoints (x, y, v) for full body
      - joints2d_lhand: 21 keypoints (x, y, v) for left hand
      - joints2d_rhand: 21 keypoints (x, y, v) for right hand
      - joints2d_face: 91 keypoints (x, y, v) for face
      - vis_face: visibility of face keypoints
      - vis_lhand: visibility of left hand keypoints
      - vis_rhand: visibility of right hand keypoints
    reference: https://github.com/openpifpaf/openpifpaf
    """

    # Run reconstruction on each tracklet
    bbox_scale = 1.0
    print(f"Running reconstruction on each tracklet...")
    bboxes = joints2d = []
    frames = []
    wb_kps = {
        "joints2d_lhand": [],
        "joints2d_rhand": [],
        "joints2d_face": [],
        "vis_face": [],
        "vis_lhand": [],
        "vis_rhand": [],
    }
    person_id_list = list(tracking_results.keys())
    
    for person_id in person_id_list:
        joints2d.extend(tracking_results[person_id]["joints2d"])
        wb_kps["joints2d_lhand"].extend(
            tracking_results[person_id]["joints2d_lhand"]
        )
        wb_kps["joints2d_rhand"].extend(
            tracking_results[person_id]["joints2d_rhand"]
        )
        wb_kps["joints2d_face"].extend(tracking_results[person_id]["joints2d_face"])
        wb_kps["vis_lhand"].extend(tracking_results[person_id]["vis_lhand"])
        wb_kps["vis_rhand"].extend(tracking_results[person_id]["vis_rhand"])
        wb_kps["vis_face"].extend(tracking_results[person_id]["vis_face"])

        frames.extend(tracking_results[person_id]["frames"])

    # Build dataset for running PyMAF-X
    dataset = Inference(
        frames=frames,
        bboxes=bboxes,
        joints2d=joints2d,
        scale=bbox_scale,
        full_body=True,
        person_ids=person_id_list,
        wb_kps=wb_kps,
        pre_load_imgs=images_array,
    )
    bboxes = dataset.bboxes
    frames = dataset.frames
    dataloader = DataLoader(dataset, batch_size=1)

    # Run PyMAF-X to infer SMPL pose parameters
    with torch.no_grad():
        pred_pose = []
        for batch in tqdm(dataloader):
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            preds_dict, _ = pymaf_model(batch)
            output = preds_dict["mesh_out"][-1]
            pred_pose.append(output["theta"][:, 13:85])

        pred_pose = torch.cat(pred_pose, dim=0)
        del batch

    pred_pose = pred_pose.cpu().numpy()
    return pred_pose
