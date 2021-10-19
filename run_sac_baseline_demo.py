import os

if __name__ == "__main__":
    txt_file = open("/home/knox/ManiSkill/available_environments.txt", "r")
    available_envs = []
    for i in txt_file.readlines():
        if "Drawer" in i and "link" in i:
            available_envs.append(i[:-1])

    for i, env in enumerate(available_envs):
        gripper_env = env[:17] + "Gripper" + env[17:]
        # train
        
        train_cmd = "python -m tools.run_rl configs/sac/sac_mani_skill_state_1M_train.py --seed=1 --cfg-options \"train_mfrl_cfg.total_steps=600000\" \"env_cfg.env_name={}\" \"rollout_cfg.type=Rollout\" \"rollout_cfg.num_procs=1\" \"eval_cfg.num_procs=1\" --gpu-ids=0".format(gripper_env)
        # eval
        eval_cmd = "python -m tools.run_rl configs/sac/sac_mani_skill_state_1M_train.py --gpu-ids=0 --cfg-options \"env_cfg.env_name={}\" \"eval_cfg.save_video=True\" \"eval_cfg.save_traj=True\" \"eval_cfg.type=Evaluation\" \"eval_cfg.num=10\" \"eval_cfg.use_log=True\" \"rollout_cfg.type=Rollout\" \"rollout_cfg.num_procs=1\" \"eval_cfg.num_procs=1\" --work-dir=./test/{} --resume-from=./work_dirs/{}/SAC/models/model_600000.ckpt --evaluation".format(gripper_env, gripper_env, gripper_env)
        # Transform pcd
        trans_cmd = "python tools/convert_state.py --max-num-traj=-1 --env-name={} --traj-name=./test/{}/SAC/test/trajectory.h5 --output-name=./test/{}/SAC/test/{}_pcd.h5 --obs-mode pointcloud".format(gripper_env, gripper_env, gripper_env, gripper_env)

        # ruN COMMAND
        os.system(train_cmd)
        os.system(eval_cmd)
        os.system(trans_cmd)
