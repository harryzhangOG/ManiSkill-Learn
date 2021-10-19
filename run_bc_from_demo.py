import os

if __name__ == "__main__":
    txt_file = open("/home/knox/ManiSkill/available_environments.txt", "r")
    available_envs = []
    for i in txt_file.readlines():
        if "Drawer" in i and "link" in i:
            available_envs.append(i[:-1])

    for i, env in enumerate(available_envs[1:2]):
        gripper_env = env[:17] + "Gripper" + env[17:]
        # train
        if False:
            print(99)

        else:
            train_cmd_no_extra = "python -m tools.run_rl configs/bc/mani_skill_point_cloud_transformer.py --gpu-ids=1 --cfg-options " \
            + "\"env_cfg.env_name={}\" \"train_mfrl_cfg.n_eval=100000\" \"eval_cfg.save_video=True\" \"train_mfrl_cfg.total_steps=2000\" ".format(gripper_env) \
            + "\"train_mfrl_cfg.init_replay_buffers=./test/{}/SAC/test/{}_pcd.h5\" --work-dir=./bc_work_dirs_tr_no_extra/{}".format(gripper_env, gripper_env, gripper_env)
            # eval
            eval_cmd_no_extra = "python -m tools.run_rl configs/bc/mani_skill_point_cloud_transformer.py --gpu-ids=0 --cfg-options " \
            + "\"env_cfg.env_name={}\" \"eval_cfg.save_video=True\" \"eval_cfg.save_traj=True\" \"eval_cfg.typr=Evaluation\" \"eval_cfg.num=100\" ".format(gripper_env) \
            + "\"eval_cfg.use_log=True\" \"rollout_cfg.type=Rollout\" \"rollout_cfg.num_procs=1\" \"eval_cfg.num_procs=1\" --work-dir=./bc_test/{}_tr_no_extra ".format(gripper_env) \
            + "--resume-from=./bc_work_dirs_tr_no_extra/{}/BC/models/model_2000.ckpt --evaluation".format(gripper_env)

            train_cmd_theta= "python -m tools.run_rl configs/bc/mani_skill_point_cloud_transformer_theta.py --gpu-ids=1 --cfg-options " \
            + "\"env_cfg.env_name={}\" \"train_mfrl_cfg.n_eval=100000\" \"eval_cfg.save_video=True\" \"train_mfrl_cfg.total_steps=2000\" ".format(gripper_env) \
            + "\"train_mfrl_cfg.init_replay_buffers=./test/{}/SAC/test/{}_pcd.h5\" --work-dir=./bc_work_dirs_tr_theta/{}".format(gripper_env, gripper_env, gripper_env)
            # eval
            eval_cmd_theta = "python -m tools.run_rl configs/bc/mani_skill_point_cloud_transformer_theta.py --gpu-ids=0 --cfg-options " \
            + "\"env_cfg.env_name={}\" \"eval_cfg.save_video=True\" \"eval_cfg.save_traj=True\" \"eval_cfg.typr=Evaluation\" \"eval_cfg.num=100\" ".format(gripper_env) \
            + "\"eval_cfg.use_log=True\" \"rollout_cfg.type=Rollout\" \"rollout_cfg.num_procs=1\" \"eval_cfg.num_procs=1\" --work-dir=./bc_test/{}_tr_theta ".format(gripper_env) \
            + "--resume-from=./bc_work_dirs_tr_theta/{}/BC/models/model_2000.ckpt --evaluation".format(gripper_env)

            train_cmd_flow = "python -m tools.run_rl configs/bc/mani_skill_point_cloud_transformer_flow.py --gpu-ids=1 --cfg-options " \
            + "\"env_cfg.env_name={}\" \"train_mfrl_cfg.n_eval=100000\" \"eval_cfg.save_video=True\" \"train_mfrl_cfg.total_steps=2000\" ".format(gripper_env) \
            + "\"train_mfrl_cfg.init_replay_buffers=./test/{}/SAC/test/{}_pcd.h5\" --work-dir=./bc_work_dirs_tr_flow/{}".format(gripper_env, gripper_env, gripper_env)
            # eval
            eval_cmd_flow = "python -m tools.run_rl configs/bc/mani_skill_point_cloud_transformer_flow.py --gpu-ids=0 --cfg-options " \
            + "\"env_cfg.env_name={}\" \"eval_cfg.save_video=True\" \"eval_cfg.save_traj=True\" \"eval_cfg.typr=Evaluation\" \"eval_cfg.num=100\" ".format(gripper_env) \
            + "\"eval_cfg.use_log=True\" \"rollout_cfg.type=Rollout\" \"rollout_cfg.num_procs=1\" \"eval_cfg.num_procs=1\" --work-dir=./bc_test/{}_tr_flow ".format(gripper_env) \
            + "--resume-from=./bc_work_dirs_tr_flow/{}/BC/models/model_2000.ckpt --evaluation".format(gripper_env)
        # ruN COMMAND
        os.system(train_cmd_no_extra)
        os.system(eval_cmd_no_extra)

        os.system(train_cmd_theta)
        os.system(eval_cmd_theta)

        os.system(train_cmd_flow)
        os.system(eval_cmd_flow)
