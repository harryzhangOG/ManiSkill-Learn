import os
available_envs = os.listdir('/home/harry/discriminative_embeddings/bc_demo_traj')
for env in available_envs:
    trans_cmd = "python tools/convert_demo_pcd.py --max-num-traj=-1 --env-name={} --traj-name=/home/harry/discriminative_embeddings/bc_demo_traj/{}/trajectory.h5 --output-name=./pm_obj_pcd_demo_traj/{}/traj_pcd.h5 --obs-mode pointcloud".format(env, env, env)
    os.system(trans_cmd)
