# Checkpoint / Restart tests on Exascale computing systems

For questions, please contact: Huihuo Zheng <huihuo.zheng@anl.gov>

Exascale computing systems often experience instabilities that can cause job terminations before completion.

To ensure large-scale simulations can continue efficiently, checkpoint/restart mechanisms are essential.

This repository provides:
	•	Simple programs to simulate common job execution issues:
(1) hanging, (2) mid-run failures, and (3) successful completion.
	•	Example submission scripts that automatically detect failures and restart jobs using healthy nodes.

The **key idea** is to over-allocate nodes, allowing jobs to be restarted on a healthy subset of nodes if a failure occurs.


## Install the package

```bash
git clone https://github.com/argonne-lcf/checkpoint_restart
cd checkpoint_restart
pip install -e .
```

## Simulation of job execution: hang, fail, success
The test_pyjob.py script allows you to simulate various job behaviors:
```bash
--hang N              # Hang for N seconds
--fail N              # Fail after N seconds
--compute T           # Compute time per iteration
--niters NITERS       # Total number of iterations
--checkpoint PATH     # Checkpoint file path
--checkpoint_time T   # Time to write a single checkpoint
```

```
python test_pyjob.py --fail 120 --checkpoint ./chkpt --niters 1000
```

## Useful scripts to compose the submission scripts that are able to handle various job execution statuses. 

- [get_healthy_nodes.sh](./get_healthy_nodes.sh) ```NODEFILE NUM_NODES_TO_SELECT NEW_NODEFILE```
  
  This script is to select a subset of healthy nodes from the entire allocation

- [check_hang.py](./check_hang.py) ```--timeout TIMEOUT --check CHECKING_PERIOD --command COMMAND --output F1:F2:F3```

  This is to constantly checking whether the job hangs or not by checking whether output files are updated or not. If it is not updated for TIMEOUT seconds. It will kill the job. 

- PBS_NODEFILE=NODEFILE [flush.sh](./flush.sh)

  This is to clean up the nodes (except the headnode, the first one on the list)


## Example submission scripts
- [qsub_multi_mpiexec.sc](./qsub_multi_mpiexec.sc)
  submission script doing continual trials of mpiexec until success or timeout

## Various simulation examples
- [fail/](./fail): job failed after 100 seconds, restart
- [hang/](./hang): job hang, kill and restart
- [success/](./success): job run seccessfully
- [resub/](./resub): job fails after 100 seconds, and restart

## Checkpoint interval optimization utility
- [optimal_checkpointing.py](./optimal_checkpointing.py)
  Determine the optimal time interval of computation between checkpoints
  for a job of determined node size and checkpointed memory per node