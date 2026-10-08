#!/bin/bash
nvidia-smi
source /data/rms/envs/python3.9_cuda11.7/bin/activate
which python
nvcc -V
pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/

pip list

model_name="model_finall"
project_root_path="/data/rms/demo"
mask_type="cartesian"
sampling_rate=20
subject=${model_name}_${mask_type}_${sampling_rate}

vis_server="192.168.1.246"
win_name=${subject}
python ./util/vis_nvidia_smi.py \
        --vis_server ${vis_server} \
        --win_name ${win_name} &

python -O ./main/main_${model_name}.py train \
                --model_name ${model_name} \
                --mask_path /data/rms/data/mask/${mask_type}_256_256_${sampling_rate}.mat \
                --project_root_path ${project_root_path}\
                --batch_size 6 \
                --dataset fastmri \
                --criterion mseloss \
                --niter 6 \
                --sparsity_loss_weight 1e-3 \
                --lr 1e-3 \
                --seed 1234 \
                --epoch 30\
                --topk 6 \
                --patch_size 8 8 \
                --num_basis 10 \
                --data_range 1 \
                --gradient_clip \
                --dataset_root /data/rms/data/fastmri/without_fat_supression \
                # --dataset_root /data/rms/data/cc \



# kill the nvidia-smi threa
pids=`ps aux | grep vis_nvidia_smi.py | grep -v grep | awk '{print $2}'`
if [ -n "$pids" ]; then
    kill -9 $pids
fi


python ./util/mail_notification.py --subject ${subject} --info_file ${project_root_path}/checkpoints/email.txt