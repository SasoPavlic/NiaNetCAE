#!/bin/bash

function get_sbatch_params() {
    # Get available nodes
    nodes=$(scontrol show nodes | grep -c NodeName)
    nodes=($(seq -s " " 0 $(($nodes - 1))))
    num_nodes=${#nodes[@]}

    # Get available CPUs
    cpus_per_node=$(scontrol show nodes | grep CPUS | awk '{print $2}')
    cpus_per_node=(${cpus_per_node//[^0-9]/})
    total_cpus=0
    for cpus in "${cpus_per_node[@]}"; do
        total_cpus=$((total_cpus + cpus))
    done

    # Get available memory
    mem_per_node=$(scontrol show nodes | grep RealMemory | awk '{print $3}')
    total_mem=0
    for mem in "${mem_per_node[@]}"; do
        mem=${mem%%[!0-9]*}
        total_mem=$((total_mem + mem))
    done

    # Get available GPUs and memory per GPU (optional)
    gpus_per_node=$(scontrol show nodes | grep Gres | awk '{print $2}' | awk -F':' '{sum += $2} END {print sum}')
    mem_per_gpu=$(scontrol show nodes | grep Gres | awk '{print $3}' | awk -F':' '{sum += $2} END {print sum}')
    total_gpus=${gpus_per_node:-0}
    total_mem_per_gpu=${mem_per_gpu:-0}

    # Set default values for missing fields
    if [[ -z "$num_nodes" ]]; then
        num_nodes=0
    fi
    if [[ -z "${cpus_per_node[0]}" ]]; then
        cpus_per_node[0]=0
    fi
    if [[ -z "$total_mem" ]]; then
        total_mem=0
    fi

    # Construct sbatch parameters
    sbatch_params=()
    sbatch_params+=("-J nianet-cae")
    sbatch_params+=("-o nianet-cae-%j.out")
    sbatch_params+=("-e nianet-cae-%j.err")
    sbatch_params+=("--nodes=1")
    sbatch_params+=("--ntasks=1")
    sbatch_params+=("--partition=gpu")
    sbatch_params+=("--time=72:00:00")
    if (( total_gpus > 0 )); then
        sbatch_params+=("--gres=gpu:${total_gpus}")
        sbatch_params+=("--mem-per-gpu=${total_mem_per_gpu}M")
    fi

    printf '%s\n' "${sbatch_params[@]}"
}

# Check if GPUs are available
gpus_available=$(scontrol show nodes | grep Gres | awk '{print $2}' | awk -F':' '{sum += $2} END {print sum}')
if (( gpus_available > 0 )); then
    # GPUs are available, submit the job
    sbatch_params=$(get_sbatch_params)
    sbatch_script="/ceph/grid/home/sasop/test.sh"  # Replace with the actual path to your sbatch script
    sbatch "${sbatch_params[@]}" "${sbatch_script}"
    if [ $? -eq 0 ]; then
        # Job submission successful, send an email notification
        echo "Subject: Job Started" | /usr/sbin/sendmail -v recipient@example.com
    fi
else
    # No GPUs available, print a message
    echo "No GPUs available. Job not submitted."
fi
