import numpy as np
import os, sys

name = "imbalance_blo_1client"
script_output_dir = f"scripts/{name}/"

tau_list = [4, 8, 16, 32]
q_list = [0.1, 0.5, 1.0]

# tau_list = [12]
# q_list = [0.1, 0.5, 1.0]

tau_list = [4, 8, 12, 16, 32]
q_list = [0.3]

if not os.path.exists(script_output_dir):
    os.makedirs(script_output_dir)
for tau in tau_list:
    for q in q_list:
        save_path = f"results/{name}/"
        out_file = f"{script_output_dir}/{tau}_{q}_frac_0.01.sh"
        if not os.path.exists(save_path + "logs"):
            os.makedirs(save_path + "logs")
        with open(out_file, mode="w", newline="\n") as script_file:
            script_file.write("#!/bin/bash -l\n")
            script_file.write("#SBATCH --nodes=1\n")
            script_file.write("#SBATCH --cpus-per-task=4\n")
            script_file.write("#SBATCH --mem=8G\n")
            script_file.write("#SBATCH --time=48:0:0\n")
            script_file.write("#SBATCH --partition=batch\n")
            script_file.write("#SBATCH --gres=gpu:1\n")
            script_file.write(f"#SBATCH --job-name={tau}_{q}_frac_0.01\n")
            script_file.write(
                f"#SBATCH --output={save_path}/logs/{tau}_{q}_frac_0.01.txt\n")
            script_file.write("export MKL_SERVICE_FORCE_INTEL=1 \n")
            script_file.write(
                f"python main_imbalance_blo.py  --epoch 1000  --round 1000 --lr 0.01 --hlr 0.02  \
--neumann 5 --inner_ep {tau}  \
--hvp_method global_batch --optim sgd  \
--output {save_path}/{tau}_{q}_frac_0.01.yaml --q_noniid {q}  \
--alpha 0.05 --beta 0.1 --frac 0.01 --momentum_rho 0.5 ")
        os.system(f"chmod +x {out_file}")
        os.system(
            f"nohup ./{out_file} > {save_path}/logs/{tau}_{q}_frac_0.01.txt 2>&1 &"
        )
