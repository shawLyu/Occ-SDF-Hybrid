import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2 as cv
import numpy as np
import os
import glob

import trimesh
import argparse
from pathlib import Path
import subprocess


scans = ["room0", "room1", "room2", "office0", "office1", "office2", "office3", "office4"]

args = argparse.ArgumentParser("Replica Evaluation")
args.add_argument("--all", action="store_true", help="Evaluate all scans")
args.add_argument("--exp_name", type=str, default="replica_mlp", help="Experiment name")

parse_args = args.parse_args()


root_dir = "../exps"
if parse_args.all:
    all_exp_name = glob.glob(os.path.join(root_dir, "replica_mlp*"))
    all_exp_name = sorted([os.path.basename(x) for x in all_exp_name])
    print(all_exp_name)
else:
    all_exp_name = [parse_args.exp_name]

# exp_name = "replica_mlp_true"

for idx, exp_name in enumerate(all_exp_name):
    out_dir = "evaluation/{}".format(exp_name)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    evaluation_txt_file = "evaluation/{}.csv".format(exp_name)
    evaluation_txt_file = open(evaluation_txt_file, 'w')
    idx = int(exp_name[-1]) - 1
    # test set
    #if not (idx in [4, 6, 7]):
    #    continue
    
    cur_root = os.path.join(root_dir, exp_name)
    print(cur_root)
    if not os.path.exists(cur_root):
        continue
    # use first timestamps
    dirs = sorted(os.listdir(cur_root))
    cur_root = os.path.join(cur_root, dirs[-1])
    files = list(filter(os.path.isfile, glob.glob(os.path.join(cur_root, "plots/*.ply"))))
    
    files.sort(key=lambda x:os.path.getmtime(x))
    ply_file = files[-1]
    print(ply_file)

    scan = scans[idx]
    # curmesh
    cull_mesh_out = os.path.join(out_dir, f"{scan}.ply")
    cmd = f"python cull_mesh.py --input_mesh {ply_file} --input_scalemat ../data/Replica/scan{idx+1}/cameras.npz --traj ../data/Replica/scan{idx+1}/traj.txt --output_mesh {cull_mesh_out}"
    print(cmd)
    os.system(cmd)

    cmd = f"python eval_recon.py --rec_mesh {cull_mesh_out} --gt_mesh ../data/Replica/cull_GTmesh/{scan}.ply"
    print(cmd)
    # accuracy_rec, completion_rec, precision_ratio_rec, completion_ratio_rec, fscore, normal_acc, normal_comp, normal_avg
    output = subprocess.check_output(cmd, shell=True).decode("utf-8")
    output = output.replace(" ", ",")
    print(output)
    
    evaluation_txt_file.write(f"{scan},{Path(ply_file).name},{output}")
    evaluation_txt_file.flush()