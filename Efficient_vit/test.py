import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
import os
import cv2
import numpy as np
import torch
from torch import nn, einsum

from sklearn.metrics import f1_score
from albumentations import Compose, RandomBrightnessContrast, \
    HorizontalFlip, FancyPCA, HueSaturationValue, OneOf, ToGray, \
    ShiftScaleRotate, ImageCompression, PadIfNeeded, GaussNoise, GaussianBlur, Rotate

from transforms.albu import IsotropicResize

from utils import get_method, check_correct, resize, shuffle_dataset, get_n_params
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from efficient_vit import EfficientViT
from utils import transform_frame
import glob
from os import cpu_count
import json
from multiprocessing.pool import Pool
from progress.bar import Bar
import pandas as pd
from tqdm import tqdm
from multiprocessing import Manager
from utils import custom_round, custom_video_round

import yaml
import argparse

import csv


MODELS_DIR = "model"
BASE_DIR = "E:\\project"
DATA_DIR = os.path.join(BASE_DIR, "crop_data")
TEST_DIR = os.path.join(DATA_DIR, "test_set")
OUTPUT_DIR = os.path.join(MODELS_DIR, "tests")

# TEST_LABELS_PATH = os.path.join(BASE_DIR, "dataset/dfdc_test_labels.csv")



# if not os.path.exists(MODELS_DIR):
#     os.makedirs(MODELS_DIR)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def create_base_transform(size):
    return Compose([
        IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
        PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
    ])

def save_roc_curves(correct_labels, preds, model_name, accuracy, loss, f1):
  plt.figure(1)
  plt.plot([0, 1], [0, 1], 'k--')

  fpr, tpr, th = metrics.roc_curve(correct_labels, preds)

  model_auc = auc(fpr, tpr)


  plt.plot(fpr, tpr, label="Model_"+ model_name + ' (area = {:.3f})'.format(model_auc))

  plt.xlabel('False positive rate')
  plt.ylabel('True positive rate')
  plt.title('ROC curve')
  plt.legend(loc='best')
  plt.savefig(os.path.join(OUTPUT_DIR, model_name +  "_" + opt.dataset + "_acc" + str(accuracy*100) + "_loss"+str(loss)+"_f1"+str(f1)+".jpg"))
  plt.clf()

def read_frames(video_path, videos, opt,config):
    
    # Get the video label based on dataset selected
    # method = get_method(video_path, DATA_DIR)
    # if "Original" in video_path:
    #     label = 0.
    # elif method == "DFDC":
    #     test_df = pd.DataFrame(pd.read_csv(TEST_LABELS_PATH))
    #     video_folder_name = os.path.basename(video_path)
    #     video_key = video_folder_name + ".mp4"
    #     label = test_df.loc[test_df['filename'] == video_key]['label'].values[0]
    # else:
    #     label = 1.
    
    # Determine label based on dataset type
    if "Celeb-real-crop" in video_path or "YouTube-real-crop" in video_path:
        label = 0
    elif "Celeb-synthesis-crop" in video_path:
        label = 1
    else:
        label = -1  # Undefined category (for debugging)

    # Ensure directory exists
    if not os.path.exists(video_path):
        print(f"Warning: Video path does not exist: {video_path}")
        return

    # Fetch frames and ensure directory is not empty
    frames_paths = sorted(os.listdir(video_path))
    if not frames_paths:
        print(f"Warning: No frames found in {video_path}")
        return

    # Calculate interval to sample frames
    frames_number = len(frames_paths)
    frames_interval = max(1, frames_number // opt.frames_per_video)

    # Group frames by face index
    frames_paths_dict = {}
    for path in frames_paths:
        for i in range(3):  # Assuming indexes are _0, _1, _2
            if f"_{i}" in path:
                frames_paths_dict.setdefault(i, []).append(path)

    # Sample frames at interval
    for key in frames_paths_dict:
        frames_paths_dict[key] = frames_paths_dict[key][::frames_interval][:opt.frames_per_video]

    # Define transformation
    transform = create_base_transform(config['model']['image-size'])

    # Store video frames
    video = {}
    for key, frames in frames_paths_dict.items():
        for frame_image in frames:
            image_path = os.path.join(video_path, frame_image)
            image = cv2.imread(image_path)

            if image is None:
                print(f"Skipping invalid image: {image_path}")
                continue  # Skip if image is not loaded

            # Apply transformations safely
            transformed = transform(image=image)
            if 'image' in transformed:
                image = transformed['image']
                if key in video:
                    video[key].append(image)
                else:
                    video[key] = [image]
            else:
                print(f"Transformation failed for {image_path}")

    # Append the processed video and label
    if video:
        videos.append((video, label, video_path))
    else:
        print(f"Skipping video {video_path} due to empty frames.")





# Main body
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--workers', default=10, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--model_path', default='', type=str, metavar='PATH',
                        help='Path to model checkpoint (default: none).')
    parser.add_argument('--dataset', type=str, default='DFDC', 
                        help="Which dataset to use (Deepfakes|Face2Face|FaceShifter|FaceSwap|NeuralTextures|DFDC)")
    parser.add_argument('--max_videos', type=int, default=-1, 
                        help="Maximum number of videos to use for training (default: all).")
    parser.add_argument('--config', type=str, 
                        help="Which configuration to use. See into 'config' folder.")
    parser.add_argument('--efficient_net', type=int, default=0, 
                        help="Which EfficientNet version to use (0 or 7, default: 0)")
    parser.add_argument('--frames_per_video', type=int, default=30, 
                        help="How many equidistant frames for each video (default: 30)")
    parser.add_argument('--batch_size', type=int, default=32, 
                        help="Batch size (default: 32)")

    opt = parser.parse_args()
    print(opt)
    
    with open(opt.config, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)

    if opt.efficient_net == 0:
        channels = 1280
    else:
        channels = 2560

    if os.path.exists(opt.model_path):
        model = EfficientViT(config=config, channels=channels, selected_efficient_net = opt.efficient_net)
        model.load_state_dict(torch.load(opt.model_path))
        model.eval()
        model = model.cuda()
    else:
        print("No model found.")
        exit()

    model_name = os.path.basename(opt.model_path)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    
    preds = []
    mgr = Manager()
    paths = []
    videos = mgr.list()

    # if opt.dataset != "DFDC":
    #     folders = ["Original", opt.dataset]
    # else:
    #     folders = [opt.dataset]

    folders = ["Celeb-real-crop", "Celeb-synthesis-crop", "YouTube-real-crop"]
    for folder in folders:
        method_folder = os.path.join(TEST_DIR, folder)  
        for index, video_folder in enumerate(os.listdir(method_folder)):
            if index == opt.max_videos:
                break
            paths.append(os.path.join(method_folder, video_folder))
      
    with Pool(processes=opt.workers) as p:
        with tqdm(total=len(paths)) as pbar:
            for v in p.imap_unordered(partial(read_frames, videos=videos, opt=opt,config=config),paths):
                pbar.update()

    video_names = np.asarray([row[2] for row in videos])
    correct_test_labels = np.asarray([row[1] for row in videos])
    videos = np.asarray([row[0] for row in videos])
    preds = []

    bar = Bar('Predicting', max=len(videos))
    
    #------------------------------------------------------------------------
        
    array = []
    #------------------------------------------------------------------------

    f = open(opt.dataset + "_" + model_name + "_labels.txt", "w+")
    for index, video in enumerate(videos):
        video_faces_preds = []
        video_name = video_names[index]
        f.write(video_name)
        for key in video:
            faces_preds = []
            video_faces = video[key]
            for i in range(0, len(video_faces), opt.batch_size):
                faces = video_faces[i:i+opt.batch_size]
                faces = torch.tensor(np.asarray(faces))
                if faces.shape[0] == 0:
                    continue
                faces = np.transpose(faces, (0, 3, 1, 2))
                faces = faces.cuda().float()
                
                pred = model(faces)
                
                scaled_pred = []
                for idx, p in enumerate(pred):
                    scaled_pred.append(torch.sigmoid(p))
                faces_preds.extend(scaled_pred)
                
            current_faces_pred = sum(faces_preds)/len(faces_preds)
            face_pred = current_faces_pred.cpu().detach().numpy()[0]
            f.write(" " + str(face_pred))
            video_faces_preds.append(face_pred)
        bar.next()
        if len(video_faces_preds) > 1:
            video_pred = custom_video_round(video_faces_preds)
        else:
            video_pred = video_faces_preds[0]
        preds.append([video_pred])
        #------------------------------------------------------------------------
        array.append([video_name,video_pred, correct_test_labels[index]])
        #------------------------------------------------------------------------
        f.write(" --> " + str(video_pred) + "(CORRECT: " + str(correct_test_labels[index]) + ")" +"\n")
        
    f.close()
    bar.finish()
    
    #------------------------------------------------------------------------
    csv_file = "predictions_.csv"
    
    # Ensure the file exists
    if not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["video","s predict", "l predict", "real"])  # Example headers

    # Read existing data
    with open(csv_file, mode='r', newline='') as file:
        reader = list(csv.reader(file))
        
    # Append new data to column 1 and 3
    for i, (name, col1, col3) in enumerate(array, start=len(reader)):
        if i < len(reader):
            reader[i][0] = name  # Update Column 1
            reader[i][1] = col1  # Update Column 2
            reader[i][3] = col3  # Update Column 4
        else:
            reader.append([name, col1, "", col3])  # Append new row
    
    # Write updated data back to file
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(reader)

    #------------------------------------------------------------------------
    loss_fn = torch.nn.BCEWithLogitsLoss()
    tensor_labels = torch.tensor([[float(label)] for label in correct_test_labels])
    tensor_preds = torch.tensor(preds)


    loss = loss_fn(tensor_preds, tensor_labels).numpy()

    #accuracy = accuracy_score(np.asarray(preds).round(), correct_test_labels)
    accuracy = accuracy_score(custom_round(np.asarray(preds)), correct_test_labels)

    f1 = f1_score(correct_test_labels, custom_round(np.asarray(preds)))
    print(model_name, "Test Accuracy:", accuracy, "Loss:", loss, "F1", f1)
    save_roc_curves(correct_test_labels, preds, model_name, accuracy, loss, f1)
