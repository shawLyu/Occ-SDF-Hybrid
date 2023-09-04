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
import pdb
import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool` is a deprecated alias')


scans = ["room0", "room1", "room2", "office0", "office1", "office2", "office3", "office4"]

args = argparse.ArgumentParser("Replica Evaluation")
args.add_argument("--all", action="store_true", help="Evaluate all scans")
args.add_argument("--exp_name", type=str, default="replica_mlp", help="Experiment name")
args.add_argument("--exp_root_dir", type=str, default="../exps", help="Results dir for all experiments")

parse_args = args.parse_args()


root_dir = parse_args.exp_root_dir
if parse_args.all:
    all_exp_name = glob.glob(os.path.join(root_dir, "replica_mlp*"))
    all_exp_name = sorted([os.path.basename(x) for x in all_exp_name])
    print(all_exp_name)
else:
    all_exp_name = [parse_args.exp_name]

# exp_name = "replica_mlp_true"

array_list_min_chamfer = []
array_list_max_f_score = []
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
    min_chamfer = 20
    max_f_score = 0
    evaluate_min_chamfer = None
    evaluate_max_f_score = None
    output_min_chamfer = None
    output_max_f_score = None
    for i in range(1, 20, 1):
        try:
            ply_file = files[-i]
        except:
            break
        ply_name = ply_file.split("/")[-1].split(".")[0]
        print(ply_file)

        scan = scans[idx]
        # curmesh
        cull_mesh_out = os.path.join(out_dir, f"{scan}_{ply_name}.ply")
        cmd = f"python cull_mesh.py --input_mesh {ply_file} --input_scalemat ../data/Replica/scan{idx+1}/cameras.npz --traj ../data/Replica/scan{idx+1}/traj.txt --output_mesh {cull_mesh_out}"
        print(cmd)
        os.system(cmd)

        cmd = f"python eval_recon.py --rec_mesh {cull_mesh_out} --gt_mesh ../data/Replica/cull_GTmesh/{scan}.ply"
        print(cmd)
        # accuracy_rec, completion_rec, precision_ratio_rec, completion_ratio_rec, fscore, normal_acc, normal_comp, normal_avg
        output = subprocess.check_output(cmd, shell=True).decode("utf-8")
        output = output.replace(" ", ",")
        numpy_output = np.fromstring(output, dtype=float, sep=',')
        chamfer = (numpy_output[0] + numpy_output[1]) / 2
        f_score = numpy_output[4]
        numpy_output = np.insert(numpy_output, 2, chamfer)
        if chamfer < min_chamfer:
            evaluate_min_chamfer = numpy_output
            output_min_chamfer = Path(ply_file).name + output
            min_chamfer = chamfer
        if f_score > max_f_score:
            evaluate_max_f_score = numpy_output
            output_max_f_score = Path(ply_file).name + output
            max_f_score = f_score

        print(output)

    array_list_max_f_score.append(evaluate_max_f_score)
    array_list_min_chamfer.append(evaluate_min_chamfer)
    print("min_chamfer", evaluate_min_chamfer)
    print("max_f_score", evaluate_max_f_score)

    evaluation_txt_file.write(f"{scan}, min_chamfer, {output_min_chamfer}")
    evaluation_txt_file.write(f"{scan}, max_f_score, {output_max_f_score}")
    evaluation_txt_file.flush()

array_list_max_f_score = np.stack(array_list_max_f_score, 0)
array_list_min_chamfer = np.stack(array_list_min_chamfer, 0)
print(array_list_max_f_score.mean(0))
print(array_list_min_chamfer.mean(0))
