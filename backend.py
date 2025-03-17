from preprocessing.detect_faces import process_videos
from preprocessing.extract_crops import extract_video
import argparse
import os
from os import cpu_count
from preprocessing.utils import get_video_paths
from multiprocessing.pool import Pool
from tqdm import tqdm
from functools import partial
import yaml
import shutil

from Efficient_vit.efficient_vit import EfficientViT
import torch
from multiprocessing import Manager
from albumentations import Compose, PadIfNeeded
from Efficient_vit.transforms.albu import IsotropicResize
import cv2
import numpy as np
from Efficient_vit.utils import custom_round, custom_video_round

from Cross_efficient_vit.cross_efficient_vit import CrossEfficientViT

from MLP.mlp import FusionMLP

video_name = None

def preprocess_videos(opt):
    """
    Preprocess videos by detecting faces, extracting crops, and processing them in parallel.

    This function handles the preprocessing of videos for both DFDC and FaceForensics datasets.
    It detects faces in videos, extracts crops, and processes them using multiprocessing.

    Parameters:
    opt (argparse.Namespace): An object containing the following attributes:
        - dataset (str): The name of the dataset ('DFDC' or 'FaceForensics').
        - data_path (str): The path to the directory containing the videos.
        - detector_type (str): The type of face detector to use.

    Returns:
    None

    Side effects:
    - Creates a 'crop' directory in the data_path.
    - Processes videos and saves face crops in the 'crop' directory.
    """
    if opt.dataset.upper() == "DFDC":
        dataset = 0
    else:
        dataset = 1
    videos_paths = []
    videos_paths = get_video_paths(opt.data_path, dataset)

    process_videos(videos_paths, opt.detector_type, dataset, opt)

    output_path = os.path.join(opt.data_path, 'crop')

    os.makedirs(output_path, exist_ok=True)
    excluded_videos = os.listdir(output_path)

    if dataset == 0:
        paths = get_video_paths(opt.data_path, dataset, excluded_videos)
    else:
        paths = get_video_paths(os.path.join(
            opt.data_path, "manipulated_sequences"), dataset)
        paths.extend(get_video_paths(os.path.join(
            opt.data_path, "original_sequences"), dataset))

    with Pool(processes=cpu_count() - 2) as p:
        with tqdm(total=len(paths)) as pbar:
            for _ in p.imap_unordered(
                partial(extract_video, root_dir=opt.data_path,
                        dataset=dataset, output_path=output_path), paths
            ):
                pbar.update()
    global video_name
    video_name = os.path.basename(videos_paths[0])
    os.remove(videos_paths[0])

def create_base_transform(size):
    """
    Create a base image transformation pipeline.

    This function creates a composition of image transformations that resize
    the image isotropically and pad it if needed to achieve a consistent size.

    Parameters:
    size (int): The target size for the image. This will be used as the maximum
                side length for resizing and the minimum height/width for padding.

    Returns:
    Compose: A composition of image transformations (IsotropicResize and PadIfNeeded)
             that can be applied to an image to standardize its size.
    """
    return Compose([
        IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA,
                        interpolation_up=cv2.INTER_CUBIC),
        PadIfNeeded(min_height=size, min_width=size,
                    border_mode=cv2.BORDER_CONSTANT),
    ])


def read_frames(video_path, opt, config):
    '''
    Reads video frames from the specified video path and returns a dictionary of frames.

    The function first calculates the interval to extract the frames based on the number of frames in the video and the desired number of frames per video. It then groups the frames with the same index to reduce the probability of skipping some faces in the same video.

    Next, the function selects only the frames at the calculated interval and limits the number of frames to the desired number of frames per video.

    Finally, the function reads the selected frames, applies a base image transformation pipeline to each frame, and returns a dictionary of the frames.

    Parameters:
    - video_path (str): The path to the video directory.
    - opt (object): An options object containing configuration parameters.
    - config (dict): A dictionary containing the model configuration.

    Returns:
    - dict: A dictionary of video frames, where the keys are the frame indices and the values are lists of transformed frame images.
    '''

    # Calculate the interval to extract the frames
    frames_number = len(os.listdir(video_path))
    frames_interval = int(frames_number / opt.frames_per_video)
    frames_paths = os.listdir(video_path)
    frames_paths_dict = {}

    # Group the faces with the same index, reduce probabiity to skip some faces in the same video
    for path in frames_paths:
        for i in range(0, 3):
            if "_" + str(i) in path:
                if i not in frames_paths_dict.keys():
                    frames_paths_dict[i] = [path]
                else:
                    frames_paths_dict[i].append(path)

    # Select only the frames at a certain interval
    if frames_interval > 0:
        for key in frames_paths_dict.keys():
            if len(frames_paths_dict) > frames_interval:
                frames_paths_dict[key] = frames_paths_dict[key][::frames_interval]

            frames_paths_dict[key] = frames_paths_dict[key][:opt.frames_per_video]

    # Select N frames from the collected ones
    video = {}
    for key in frames_paths_dict.keys():
        for index, frame_image in enumerate(frames_paths_dict[key]):
            # image = np.asarray(resize(cv2.imread(os.path.join(video_path, frame_image)), IMAGE_SIZE))
            transform = create_base_transform(config['model']['image-size'])
            image = transform(image=cv2.imread(
                os.path.join(video_path, frame_image)))['image']
            if len(image) > 0:
                if key in video:
                    video[key].append(image)
                else:
                    video[key] = [image]
    return video


def test_branch(opt, branch_config, model):
    paths = None
    video = []
    method_folder = os.path.join(opt.data_path, 'crop')

    for index, video_folder in enumerate(os.listdir(method_folder)):
        paths = os.path.join(method_folder, video_folder)

    video = read_frames(video_path=paths, opt=opt, config=branch_config)

    video_faces_preds = []

    for key in video:
        faces_preds = []
        video_faces = video[key]

        for i in range(0, len(video_faces), opt.batch_size):
            faces = video_faces[i:i + opt.batch_size]
            if len(faces) == 0:  # Skip empty batches
                continue

            # Convert to NumPy, transpose, then to Torch tensor
            faces = np.array(faces).transpose((0, 3, 1, 2))
            faces = torch.from_numpy(faces).cuda().float()

            pred = model(faces)
            scaled_pred = [torch.sigmoid(p) for p in pred]
            faces_preds.extend(scaled_pred)

        if faces_preds:  # Ensure there are predictions
            current_faces_pred = sum(faces_preds) / len(faces_preds)
            video_faces_preds.append(
                current_faces_pred.cpu().detach().numpy()[0])

    if len(video_faces_preds) > 1:
        video_pred = custom_video_round(video_faces_preds)
    else:
        video_pred = video_faces_preds[0]
    return video_pred


def s_branch(opt):

    with open(opt.config_sbranch, 'r') as ymlfile:
        config_sbranch = yaml.safe_load(ymlfile)

    if opt.efficient_net == 0:
        channels = 1280
    else:
        channels = 2560

    if os.path.exists(opt.model_path_sbranch):
        model = EfficientViT(config=config_sbranch, channels=channels,
                             selected_efficient_net=opt.efficient_net)
        model.load_state_dict(torch.load(opt.model_path_sbranch))
        model.eval()
        model = model.cuda()
    else:
        print("No model found.")
        exit()

    return test_branch(opt, config_sbranch, model)


def l_branch(opt):
    with open(opt.config_lbranch, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)

    if os.path.exists(opt.model_path_lbranch):
        model = CrossEfficientViT(config=config)
        model.load_state_dict(torch.load(opt.model_path_lbranch))
        model.eval()
        model = model.cuda()
    else:
        print("No model found.")
        exit()

    return test_branch(opt, config, model)


def predict():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="DFDC", type=str,
                        help='Dataset (DFDC / FACEFORENSICS)')
    parser.add_argument('--data_path', default='crop_data\\backend_test',
                        type=str, help='Videos directory')
    parser.add_argument("--detector_type", help="Type of the detector", default="FacenetDetector",
                        choices=["FacenetDetector"])
    parser.add_argument(
        "--processes", help="Number of processes", default=1, type=int)
    parser.add_argument('--workers', default=10, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--model_path_sbranch', default='Efficient_vit\\efficient_vit.pth', type=str, metavar='PATH',
                        help='Path to S branch model checkpoint (default: none).')
    parser.add_argument('--model_path_lbranch', default='Cross_efficient_vit\\cross_efficient_vit.pth', type=str, metavar='PATH',
                        help='Path to L branch model checkpoint (default: none).')
    parser.add_argument('--max_videos', type=int, default=-1,
                        help="Maximum number of videos to use for training (default: all).")
    parser.add_argument('--config_sbranch', type=str, default='Efficient_vit\\configs\\architecture.yaml',
                        help="Which configuration to use for S branch. See into 'config' folder.")
    parser.add_argument('--config_lbranch', type=str, default='Cross_efficient_vit\\configs\\architecture.yaml',
                        help="Which configuration to use for L branch. See into 'config' folder.")
    parser.add_argument('--efficient_net', type=int, default=0,
                        help="Which EfficientNet version to use (0 or 7, default: 0)")
    parser.add_argument('--frames_per_video', type=int, default=30,
                        help="How many equidistant frames for each video (default: 30)")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size (default: 32)")

    opt = parser.parse_args()
    print(opt)

    preprocess_videos(opt=opt)

    s_prediction = s_branch(opt=opt)
    print("S Branch prediction: ", s_prediction)

    l_prediction = l_branch(opt=opt)
    print("L Branch prediction: ", l_prediction)
    
    mlp= FusionMLP()
    mlp.load_state_dict(torch.load("MLP\\fusion_mlp.pth"))
    mlp.eval()
    
    #  Convert NumPy float32 to PyTorch Tensor
    s_prediction = torch.tensor([[s_prediction]])  # Shape: (1,1)
    l_prediction = torch.tensor([[l_prediction]])  # Shape: (1,1)
    
    input_data = torch.cat((s_prediction, l_prediction), dim=1)
    final_prediction = mlp(input_data)
    
    print(f"Final Prediction: {final_prediction.item():.4f}")  # Closer to 1 → Real, Closer to 0 → Fake

    boxes = os.path.join(opt.data_path, "boxes")
    crop = os.path.join(opt.data_path, "crop")
    
    
    for file in [boxes, crop]:
        if os.path.exists(file):
            shutil.rmtree(file)

    result = {
        "video_name": video_name,
        "s_prediction": s_prediction.item(), 
        "l_prediction": l_prediction.item(), 
        "final_prediction": final_prediction.item()
    }
    
    return result

if __name__ == "__main__":
    predict()
