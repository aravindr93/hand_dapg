DEFAULT_CONFIG = dict(
  lr= 0.0003,
  delta= 0.01,
  traj_budget_expert= 12500,
  use_cuda= True,
  num_files_val= 1,
  camera_name= "view_2",
  seed= 1000,
  num_files_train= 1,
  batch_size_viz_pol= 128,
  id_post= "view_2_public_best_20trag_10ep",
  dagger_epoch= 10,
  env_id= "mjrl_pen_reposition-v2",
  trainer_epochs= 10,
  val_traj_per_file= 5,
  viz_policy_folder_dagger= "dagger_hand_pen_view_2_viz_policy",
  device_id= 0,
  expert_policy_folder= "hand_pen_expert",
  train_traj_per_file= 20,
  bc_epoch= 20,
  env_name= "hand_pen",
  beta_start= 1,
  has_robot_info= True,
  gen_traj_dagger_ep= 20,
  use_late_fusion= True,
  use_tactile= True,
  sliding_window= 80,
  train_expert= False,
  eval_num_traj= 100,
  horizon_il= 150,
  beta_decay= 0.2,
  num_traj_expert= 50
)
