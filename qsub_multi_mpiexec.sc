#!/bin/bash
#PBS -l walltime=0:20:00
#PBS -q lustre_scaling
#PBS -N test_multi_mpiexec
#PBS -l select=4
#PBS -A Aurora_deployment

MAX_TRIALS=10

IFS='.' read -ra ADDR <<< "$PBS_JOBID"
export JOBID=$ADDR

module load frameworks
cd ${PBS_O_WORKDIR}

cat $PBS_NODEFILE | uniq > nodefile_all

export JOBSIZE=2

rm -f check_hang.r$JOBID

echo "Started running job at `date`"

for RUN in `seq 1 $MAX_TRIALS`
do
    # select a subset of nodes to run the job
    get_healthy_nodes.sh $PBS_NODEFILE $JOBSIZE pbs_nodefile$RUN
    export PBS_NODEFILE=pbs_nodefile$RUN

    # constantly check the job and kill the job if it hangs for 300 seconds
    check_hang.py --timeout 300 --outputs $PBS_JOBNAME.o$JOBID:$PBS_JOBNAME.e$JOBID:output.log --kill-command "pkill -u $USER python ./test_pyjob.py" >> check_hang.r$JOBID &

    # constantly check the job and kill the job if it generates nan/inf
    check_nan.py --outputs $PBS_JOBNAME.o$JOBID:$PBS_JOBNAME.e$JOBID:output.log --kill-command "pkill -u $USER python ./test_pyjob.py" >> check_nan.r$JOBID &

    # run the actual job, in this case, the job will run for 200 seconds and fail (finished about 9 iterations each time)
    mpiexec -np $((JOBSIZE*12)) --ppn 12 launcher.sh python ./test_pyjob.py --compute 10 --niters 100 --output output.log

    EXIT_CODE=$?
    # Check the job status
    if [ $EXIT_CODE -ne 0 ]; then
	    echo "Job exited with $EXIT_CODE error code; will rerun"	
        # We might want to do more things here. 
    else
	    echo "Job run successfully"
	    break
    fi
    echo "Rerun the job at `date`; time of trials: $RUN"
    # clear up the nodes for rerun the job
    pkill python
    PBS_NODEFILE=nodefile_all flush.sh
    sleep 5
done

echo "Finished running jobs with $RUN trials at `date`"
