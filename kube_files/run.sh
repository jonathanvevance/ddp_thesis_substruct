# Note: all paths referenced here are relative to the Docker container.

# Add the Nvidia drivers to the path
export PATH="/usr/local/nvidia/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/scratch/scratch6/jonathanvevance/envs/ddp_thesis/lib:$LD_LIBRARY_PATH"

# Tools config for CUDA, Anaconda installed in the common /tools directory
source /tools/config.sh

# Activate your environment
source activate /scratch/scratch6/jonathanvevance/envs/ddp_thesis/

# Change to the directory in which your code is present
cd /scratch/scratch6/jonathanvevance/projects/ddp_thesis_substruct/

# Run the code. The -u option is used here to use unbuffered writes
# so that output is piped to the file as and when it is produced
python -u ./src/train_pairwise.py &>> /scratch/scratch6/jonathanvevance/projects/ddp_thesis_substruct/kube_files/logs.txt  
# python -u ./kube_files/torch_debug.py &>> /scratch/scratch6/jonathanvevance/projects/ddp_thesis_substruct/kube_files/logs.txt


