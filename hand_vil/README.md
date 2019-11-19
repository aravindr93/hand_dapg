# hand_vil
Learning Deep Visuomotor Policies for Dexterous Hand Manipulation

## Setup

Each repository above contains detailed setup instructions. 
1. **Step 1:** Install [mjrl](https://github.com/aravindr93/mjrl), using instructions in the repository ([direct link](https://github.com/aravindr93/mjrl/tree/master/setup)). `mjrl` comes with an anaconda environment which helps to easily import and use a variety of MuJoCo tasks.
2. **Step 2:** Install [mj_envs](https://github.com/vikashplus/mj_envs) by following the instructions in the repository. Note that `mj_envs` uses git submodules, and hence must be cloned correctly per instructions in the repo.
3. **Step 3:** After setting up `mjrl` and `mj_envs`, Add them to your python path alongside `hand_vil`.
```
$ export PYTHONPATH=$PYTHONPATH:<your_path>/mjrl
$ export PYTHONPATH=$PYTHONPATH:<your_path>/mj_envs
$ export PYTHONPATH=$PYTHONPATH:<your_path>/hand_dapg/hand_vil
```

## Training the Visuomotor policies

1. **Step 1:** Make a "local_settings.py" file and set the variable "MAIN_DIR" to point to the root folder of the project. Consult local_settings.py.sample.

3. **Step 3** We already have the expert policies for each of the environments fetched from [hand_dapg](https://github.com/aravindr93/hand_dapg). So we are ready to train the visual policy for any of the above 4 environments.
* It is highly reccomended that you use a machine with a GPU for faster training. If you are not planning on using a GPU, make sure to set `use_cuda` in the config to False.
* All the training for the different environments are present in configs/
* Move the config that you want to run to the root project directory. For example to use the Hand Hammer config run the following command:
```
mv configs/config_main_hammer.py config_main.py
```
* Now, we are ready the train the visual model.
```
$ python run.py
```

Note that this will save the generated training data to `gen_data/data/<name_of_run>/train_data`
and will save the generated validation data to `gen_data/data/<name_of_run>/val_data`, and the trained policy
to `gen_data/data/<name_of_run>/<abbr_run_name>_viz_policy`,

## Visualizing a trained policy
Once you have the trained policy you can set the appropriate `CAMERA_NAME`,  `ENV_NAME`, `VIZ_FOLDER` and `FULL_POLICY_PATH` (the trained policies are going to be saved in the `gen_data` folder) in the `viz_policy.py`. This will save multiple episodes of the trained policy being unrolled under `gen_data/results/vides/VIZ_FOLDER`.

## Bibliography

If you use the code in this or associated repositories above, please cite the following paper.
```
@INPROCEEDINGS{Jain-ICRA-19,
    AUTHOR    = {Divye Jain AND Andrew Li AND Shivam Singhal AND 
                 Aravind Rajeswaran AND Vikash Kumar AND Emanuel Todorov},
    TITLE     = "{Learning Deep Visuomotor Policies for Dexterous Hand Manipulation}",
    BOOKTITLE = {International Conference on Robotics and Automation (ICRA)},
    YEAR      = {2019},
}
```
