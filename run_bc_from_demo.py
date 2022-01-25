import os

if __name__ == "__main__":
    txt_file = open("/home/harry/discriminative_embeddings/third_party/ManiSkill/available_environments.txt", "r")
    available_envs = []
    for i in txt_file.readlines():
        if "Drawer" in i and "link" in i:
            available_envs.append(i[:-1])

    for i, env in enumerate(available_envs[:1]):
        gripper_env = env[:17] + "Gripper" + env[17:]
        if '1000' in gripper_env:

            train_cmd_no_extra = "python -m tools.run_rl configs/bc/mani_skill_point_cloud_transformer.py --gpu-ids=1 --cfg-options " \
            + "\"env_cfg.env_name={}\" \"train_mfrl_cfg.n_eval=100000\" \"eval_cfg.save_video=True\" \"train_mfrl_cfg.total_steps=5000\" ".format(gripper_env) \
            + "\"train_mfrl_cfg.init_replay_buffers=/home/knox/Downloads/the_dataset/test/{}/SAC/test/{}_pcd.h5\" --work-dir=./bc_work_dirs_tr_no_extra/{}".format(gripper_env, gripper_env, gripper_env)
            # eval
            eval_cmd_no_extra = "python -m tools.run_rl configs/bc/mani_skill_point_cloud_transformer.py --gpu-ids=0 --cfg-options " \
            + "\"env_cfg.env_name={}\" \"eval_cfg.save_video=True\" \"eval_cfg.save_traj=True\" \"eval_cfg.typr=Evaluation\" \"eval_cfg.num=100\" ".format(gripper_env) \
            + "\"eval_cfg.use_log=True\" \"rollout_cfg.type=Rollout\" \"rollout_cfg.num_procs=1\" \"eval_cfg.num_procs=1\" --work-dir=./bc_test_batch/{}_tr_no_extra ".format(gripper_env) \
            + "--resume-from=./bc_work_dirs_tr_no_extra/{}/BC/models/model_5000.ckpt --evaluation".format(gripper_env)

            train_cmd_theta= "python -m tools.run_rl configs/bc/mani_skill_point_cloud_transformer_theta.py --gpu-ids=1 --cfg-options " \
            + "\"env_cfg.env_name={}\" \"train_mfrl_cfg.n_eval=100000\" \"eval_cfg.save_video=True\" \"train_mfrl_cfg.total_steps=5000\" ".format(gripper_env) \
            + "\"train_mfrl_cfg.init_replay_buffers=/home/knox/Downloads/the_dataset/test/{}/SAC/test/{}_pcd.h5\" --work-dir=./bc_work_dirs_tr_theta/{}".format(gripper_env, gripper_env, gripper_env)
            # eval
            eval_cmd_theta = "python -m tools.run_rl configs/bc/mani_skill_point_cloud_transformer_theta.py --gpu-ids=0 --cfg-options " \
            + "\"env_cfg.env_name={}\" \"eval_cfg.save_video=True\" \"eval_cfg.save_traj=True\" \"eval_cfg.typr=Evaluation\" \"eval_cfg.num=100\" ".format(gripper_env) \
            + "\"eval_cfg.use_log=True\" \"rollout_cfg.type=Rollout\" \"rollout_cfg.num_procs=1\" \"eval_cfg.num_procs=1\" --work-dir=./bc_test_batch/{}_tr_theta ".format(gripper_env) \
            + "--resume-from=./bc_work_dirs_tr_theta/{}/BC/models/model_5000.ckpt --evaluation".format(gripper_env)

            train_cmd_flow = "python -m tools.run_rl configs/bc/mani_skill_point_cloud_transformer_flow.py --gpu-ids=1 --cfg-options " \
            + "\"env_cfg.env_name={}\" \"train_mfrl_cfg.n_eval=100000\" \"eval_cfg.save_video=True\" \"train_mfrl_cfg.total_steps=5000\" ".format(gripper_env) \
            + "\"train_mfrl_cfg.init_replay_buffers=/home/knox/Downloads/the_dataset/test/{}/SAC/test/{}_pcd.h5\" --work-dir=./bc_work_dirs_tr_flow/{}".format(gripper_env, gripper_env, gripper_env)
            # eval
            eval_cmd_flow = "python -m tools.run_rl configs/bc/mani_skill_point_cloud_transformer_flow.py --gpu-ids=0 --cfg-options " \
            + "\"env_cfg.env_name={}\" \"eval_cfg.save_video=True\" \"eval_cfg.save_traj=True\" \"eval_cfg.typr=Evaluation\" \"eval_cfg.num=100\" ".format(gripper_env) \
            + "\"eval_cfg.use_log=True\" \"rollout_cfg.type=Rollout\" \"rollout_cfg.num_procs=1\" \"eval_cfg.num_procs=1\" --work-dir=./bc_test_batch/{}_tr_flow ".format(gripper_env) \
            + "--resume-from=./bc_work_dirs_tr_flow/{}/BC/models/model_5000.ckpt --evaluation".format(gripper_env)
            # ruN COMMAND
            # os.system(train_cmd_no_extra)
            os.system(eval_cmd_no_extra)

            # os.system(train_cmd_theta)
            os.system(eval_cmd_theta)

            # os.system(train_cmd_flow)
            os.system(eval_cmd_flow)

            # extract csv
            import csv
            no_extra_succ = []
            with open("./bc_test_batch/{}_tr_no_extra/BC/test/statistics.csv".format(gripper_env), 'r') as f:
                s = csv.reader(f, delimiter=',')
                next(s)
                for l in s:
                    no_extra_succ.append(float(l[-1]))
            no_extra_succ_rate = sum(no_extra_succ) / len(no_extra_succ)
            f.close()

            dtheta_succ = []
            with open("./bc_test_batch/{}_tr_theta/BC/test/statistics.csv".format(gripper_env), 'r') as f:
                s = csv.reader(f, delimiter=',')
                next(s)
                for l in s:
                    dtheta_succ.append(float(l[-1]))
            dtheta_succ_rate = sum(dtheta_succ) / len(dtheta_succ)
            f.close()

            flow_succ = []
            with open("./bc_test_batch/{}_tr_flow/BC/test/statistics.csv".format(gripper_env), 'r') as f:
                s = csv.reader(f, delimiter=',')
                next(s)
                for l in s:
                    flow_succ.append(float(l[-1]))
            flow_succ_rate = sum(flow_succ) / len(flow_succ)
            f.close()

            result = {}
            with open("bc_baseline_result_drawer.txt", "a") as txtfile:
                    txtfile.write(gripper_env + ": %f, %f, %f"%(no_extra_succ_rate, dtheta_succ_rate, flow_succ_rate) + "\n")
            txtfile.close()
