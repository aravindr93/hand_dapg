# Job scripts

Here we provide easy job scripts for running training algorithms on the hand manipulation tasks (in general any gym environment). To run the experiments, use the commands below. The experiments are run through the job script provided which tasks two arguments:
- `output`: path to directory where all the results will be saved
- `config`: a config `.txt` file with all the experiment parameters (examples are provided)
The script has to be run from this directory, i.e. `hand_dapg/dapg/examples` 

1. To train an NPG policy from scratch
```
$ python job_script.py --output rl_scratch_exp --config rl_scratch.txt
```
In this case, we don't give the algorithm any demonstration data.

2. To train policy with behavior cloning initialization followed by NPG finetuning, run
```
$ python job_script.py --output bcrl_exp --config bcrl.txt
```

3. To train policy with DAPG
```
$ python job_script.py --output dapg_exp --config dapg.txt
```

The most critical parts of the `config` are:
- `algorithm`: where we specify either NPG, BCRL, or DAPG
- `demo_file`: path to the demo pickle file (None to not provide any demos)

Based on the above, we can run all the three algorithms with a common backbone.