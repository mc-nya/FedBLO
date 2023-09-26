import numpy as np
import os, sys

name = "imbalance_blo_no_momentum"
script_output_dir = f"scripts/{name}/"

# mctest : q=0.1, result/
# mc1 : q=0.3, result/
# mc2 : q=0.5, result/
# mc3: q=0.1, result2/
# mc4: q=0.3, result2/
# mc5: q=0.5, result2/
server_name = "mc3"
# assign q list and result path according to server_name
if server_name == "mctest":
    q_list = [0.1]
    result_path = "results/"
elif server_name == "mc1":
    q_list = [0.3]
    result_path = "results/"
elif server_name == "mc2":
    q_list = [0.5]
    result_path = "results/"
elif server_name == "mc3":
    q_list = [0.1]
    result_path = "results2/"
elif server_name == "mc4":
    q_list = [0.3]
    result_path = "results2/"
elif server_name == "mc5":
    q_list = [0.5]
    result_path = "results2/"
else:
    raise ValueError("server_name not recognized")

tau_list = [4, 8, 12]
# q_list = [0.1]

if not os.path.exists(script_output_dir):
    os.makedirs(script_output_dir)
for tau in tau_list:
    for q in q_list:
        save_path = f"{result_path}/{name}/"
        out_file = f"{script_output_dir}/{tau}_{q}_frac_0.1.sh"
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
            script_file.write(f"#SBATCH --job-name={tau}_{q}_frac_0.1\n")
            script_file.write(
                f"#SBATCH --output={save_path}/logs/{tau}_{q}_frac_0.1.txt\n")
            script_file.write("export MKL_SERVICE_FORCE_INTEL=1 \n")
            script_file.write(
                f"python main_imbalance_blo.py  --epoch 1000  --round 1000 --lr 0.01 --hlr 0.02  \
--neumann 5 --inner_ep {tau}  \
--hvp_method global_batch --optim sgd  \
--output {save_path}/{tau}_{q}_frac_0.1.yaml --q_noniid {q}  \
--alpha 0.05 --beta 0.1 --frac 0.1 --momentum_rho 1.0 --sarah_momentum 0.5 ")

        os.system(f"chmod +x {out_file}")
        os.system(
            f"nohup ./{out_file} > {save_path}/logs/{tau}_{q}_frac_0.1.txt 2>&1 &"
        )
