import os
import torch
import random
import argparse
import datetime
import numpy as np

from models.coin import TD3
from utils import set_env
from coin_replay_buffer import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3", help='Policy name (TD3)')
    parser.add_argument("--env_name", default="intersection", help='MetaDrive environment name')
    parser.add_argument("--seed", default=0, type=int, help='Sets Gym, PyTorch and Numpy seeds')
    parser.add_argument("--buffer_size", default=1e7, type=int, help='Size of the experience replay buffer')
    parser.add_argument('--start_timesteps', default=3e4, type=int, help='Time steps initial random policy is used')
    parser.add_argument("--max_timesteps", default=5e5, type=int, help='Max time steps to run environment')
    parser.add_argument("--hidden_size", default=256, type=int, help='The number of hidden units')
    parser.add_argument("--expl_noise", default=0.05, type=float, help='Std of Gaussian exploration noise')
    parser.add_argument("--batch_size", default=256, type=int, help='Batch size for both actor and critic')
    parser.add_argument("--discount", default=0.99, type=float, help='Discount factor')
    parser.add_argument("--tau", default=0.005, type=float, help='Target network update rate')  #
    parser.add_argument("--actor_lr", default=1e-4, type=float, help='Learning rate for actor network')  #
    parser.add_argument("--critic_lr", default=2e-4, type=float, help='Learning rate for critic network')  #
    parser.add_argument("--policy_noise", default=0.1, help='Noise added to target policy during critic update')  #
    parser.add_argument("--noise_clip", default=0.5, help='Range to clip target policy noise')  #
    parser.add_argument("--policy_freq", default=3, type=int, help='Frequency of delayed policy updates')  #
    parser.add_argument("--weight_fac", default=0.8, type=float, help='Weight factor for individual and global objectives')
    parser.add_argument("--save_model", default=True, help='Save model and optimizer parameters')  #
    parser.add_argument("--save_freq", default=5e3, type=int, help='How often (time steps) we save model')  #
    parser.add_argument("--load_model", default="", help='')  # Model load file name, "" doesn't load, "default" uses file_name
    args = parser.parse_args()

    file_name = "{}_{}_{}".format(args.policy, args.env_name, args.seed)
    exp_path = './runs/{}_{}_{}/'.format(args.env_name.lower(), datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S"), args.seed)
    models_path = exp_path + "/model/"
    trains_path = exp_path + "/train/"
    results_path = exp_path + "/result/"
    gifs_path = exp_path + '/gifs/'
    print("---------------------------------------------")
    print("Policy: {}, Environment: {}, Seed: {}".format(args.policy, args.env_name, args.seed))
    print("---------------------------------------------")

    # Create exp path
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    if args.save_model and not os.path.exists(models_path):
        os.makedirs(models_path)
    if not os.path.exists(trains_path):
        os.makedirs(trains_path)
    if not os.path.exists(gifs_path):
        os.makedirs(gifs_path)

    # Set summary
    summary = SummaryWriter(trains_path)
    summary.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Set environment
    env = set_env(args.env_name, args.seed)
    state_dim = 10
    max_num_agents = int(env.config['num_agents']) + 10
    obs_dim = env.observation_space['agent0'].shape[0]
    action_dim = env.action_space['agent0'].shape[0]
    max_action = float(env.action_space['agent0'].high[0])

    # Set agent
    kwargs = {
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "state_dim": state_dim,
        "max_action": max_action,
        "tau": args.tau,
        "discount": args.discount,
        "actor_lr": args.actor_lr,
        "critic_lr": args.critic_lr
    }
    # Initialize policy
    if args.policy == "TD3":
        # Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = float(args.policy_noise) * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        kwargs["hidden_dim"] = args.hidden_size
        kwargs["weight_fac"] = args.weight_fac
        policy = TD3(**kwargs)
    else:
        raise NotImplementedError
    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(models_path + "/{}".format(policy_file))

    # Set experience buffer
    replay_buffer = ReplayBuffer(obs_dim, state_dim, action_dim, max_num_agents, int(args.buffer_size))

    # Training loop
    obs_n, _ = env.reset()

    episode_reward = 0
    episode_global_reward = 0
    episode_timesteps = 0
    episode_num = 0
    episode_crash_veh = 0
    episode_arr_dest = 0
    episode_run_out = 0
    episode_veh_step_dict = dict()
    episode_veh_finish_step_dict = dict()
    num_updates = 0
    best_arr_dest = 0
    for t in range(int(args.max_timesteps)):
        episode_timesteps += 1

        # Get state info for all agents
        s_n = dict()
        for agent in list(env.agents.keys()):
            checkpoint1, checkpoint2 = env.agents[agent].navigation.get_checkpoints()
            single_state = [env.agents[agent].position[0] / 150,
                            env.agents[agent].position[1] / 150,
                            env.agents[agent].velocity_km_h[0] / env.agents[agent].max_speed_km_h,
                            env.agents[agent].velocity_km_h[1] / env.agents[agent].max_speed_km_h,
                            env.agents[agent].heading[0],
                            env.agents[agent].heading[1],
                            checkpoint1[0] / 150,
                            checkpoint1[1] / 150,
                            checkpoint2[0] / 150,
                            checkpoint2[1] / 150]
            s_n[agent] = single_state

        # Retrieve the global state for all agents
        global_s_n = dict()
        global_mask_n = dict()
        for agent in list(env.agents.keys()):
            global_s_n[agent] = [s_n[agent]]
            global_mask_n[agent] = [0]
            for other_agent in list(env.agents.keys()):
                if other_agent != agent:
                    global_s_n[agent].append(s_n[other_agent])
                    global_mask_n[agent].append(0)
                else:
                    pass

            # Padding global state for all agents
            padding_dim = max_num_agents - len(list(env.agents.keys()))
            for _ in range(padding_dim):
                global_s_n[agent].append(np.zeros(state_dim).tolist())
                global_mask_n[agent].append(1)

            # Convert to numpy array
            global_s_n[agent] = np.array(global_s_n[agent])
            global_mask_n[agent] = np.array(global_mask_n[agent])

        a_n = dict()
        # Select action randomly or according to policy
        if t < args.start_timesteps:
            for agent, obs in obs_n.items():
                a_n[agent] = env.action_space[agent].sample()
        else:
            for agent, obs in obs_n.items():
                global_s, global_mask = global_s_n[agent], global_mask_n[agent]
                a_n[agent] = (policy.select_action(obs) +
                              np.random.normal(0, max_action * args.expl_noise, size=action_dim)).clip(-max_action, max_action)

        # Retrieve the global action for all agents
        global_a_n = dict()
        for agent in list(env.agents.keys()):
            global_a_n[agent] = [a_n[agent].tolist()]
            for other_agent in list(env.agents.keys()):
                if other_agent != agent:
                    global_a_n[agent].append(a_n[other_agent].tolist())
                else:
                    pass
            # Padding global state for all agents
            padding_dim = max_num_agents - len(list(env.agents.keys()))
            for _ in range(padding_dim):
                global_a_n[agent].append(np.zeros(action_dim).tolist())

            global_a_n[agent] = np.array(global_a_n[agent])

        # Perform action
        next_obs_n, r_n, d_n, _, info = env.step(a_n)

        # Get the next state info for all agents
        next_s_n = dict()
        for agent in list(next_obs_n.keys()):
            checkpoint1, checkpoint2 = info[agent]['checkpoints']
            single_state = [info[agent]['position'][0] / 150,
                            info[agent]['position'][1] / 150,
                            info[agent]['velocity_km_h'][0] / info[agent]['max_speed_km_h'],
                            info[agent]['velocity_km_h'][1] / info[agent]['max_speed_km_h'],
                            info[agent]['heading'][0],
                            info[agent]['heading'][1],
                            checkpoint1[0] / 150,
                            checkpoint1[1] / 150,
                            checkpoint2[0] / 150,
                            checkpoint2[1] / 150]
            next_s_n[agent] = single_state

        # Retrieve next global state for all agents
        next_global_s_n = dict()
        next_global_mask_n = dict()
        for agent in list(next_obs_n.keys()):
            next_global_s_n[agent] = [next_s_n[agent]]
            next_global_mask_n[agent] = [0]
            for other_agent in list(next_obs_n.keys()):
                if other_agent != agent:
                    next_global_s_n[agent].append(next_s_n[other_agent])
                    next_global_mask_n[agent].append(0)
            # Padding the next global state
            padding_dim = max_num_agents - len(list(next_obs_n.keys()))
            for _ in range(padding_dim):
                next_global_s_n[agent].append(np.zeros(state_dim).tolist())
                next_global_mask_n[agent].append(1)

            next_global_s_n[agent] = np.array(next_global_s_n[agent])
            next_global_mask_n[agent] = np.array(next_global_mask_n[agent])

        # Retrieve the next action for all agents
        next_a_n = dict()
        for agent, next_obs in next_obs_n.items():
            next_a_n[agent] = (policy.select_action(next_obs) +
                               np.random.normal(0, max_action * args.expl_noise, size=action_dim)).clip(-max_action, max_action)

        # Calculate the next actions for all agents
        next_global_a_n = dict()
        for agent in list(next_obs_n.keys()):
            next_global_a_n[agent] = [next_a_n[agent].tolist()]
            for other_agent in list(next_obs_n.keys()):
                if other_agent != agent:
                    next_global_a_n[agent].append(next_a_n[other_agent].tolist())
            padding_dim = max_num_agents - len(list(next_obs_n.keys()))
            for _ in range(padding_dim):
                next_global_a_n[agent].append(np.zeros(action_dim).tolist())

            next_global_a_n[agent] = np.array(next_global_a_n[agent])

        done_bool_n = dict()
        neighbor_r_n = dict()
        for agent, done in d_n.items():
            if agent != '__all__':
                done_bool_n[agent] = float(done) if episode_timesteps < env.config['horizon'] else 0

        # Construct global reward/objective
        global_reward = np.mean(list(r_n.values()))
        episode_global_reward += np.mean(list(r_n.values()))

        for agent in list(obs_n.keys()):
            # Store data in replay buffer
            replay_buffer.add(obs_n[agent], a_n[agent], next_obs_n[agent], r_n[agent], global_reward,
                              done_bool_n[agent], global_s_n[agent], global_a_n[agent], global_mask_n[agent],
                              next_global_s_n[agent], next_global_a_n[agent], next_global_mask_n[agent])
            # Remove the observations of finished agents
            if d_n[agent]:
                del next_obs_n[agent]
                # Count the trajectory length of vehicles that not collide
                if info[agent]['arrive_dest']:
                    episode_veh_finish_step_dict[agent] = episode_veh_step_dict[agent]
                else:
                    episode_veh_finish_step_dict[agent] = env.config['horizon']

            # Record episode metrics
            if agent not in episode_veh_step_dict:
                episode_veh_step_dict[agent] = 1
            else:
                episode_veh_step_dict[agent] += 1
            if info[agent]['crash_vehicle']:
                episode_crash_veh += 1
            if info[agent]['out_of_road']:
                episode_run_out += 1
            if info[agent]['arrive_dest']:
                episode_arr_dest += 1

        obs_n = next_obs_n
        episode_reward += sum(r_n.values())

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            q_i, q_g, q_v, q_i_l, q_g_l, q_v_l, vae_l = policy.train(replay_buffer, args.batch_size)
            summary.add_scalar(tag='Train/VAE Loss',
                               scalar_value=vae_l,
                               global_step=num_updates)
            summary.add_scalar(tag='Train/Individual Q Loss',
                               scalar_value=q_i_l,
                               global_step=num_updates)
            summary.add_scalar(tag='Train/Global Q Loss',
                               scalar_value=q_g_l,
                               global_step=num_updates)
            summary.add_scalar(tag='Train/Global V Loss',
                               scalar_value=q_v_l,
                               global_step=num_updates)
            summary.add_scalar(tag='Train/Individual Q',
                               scalar_value=q_i,
                               global_step=num_updates)
            summary.add_scalar(tag='Train/Global Q',
                               scalar_value=q_g,
                               global_step=num_updates)
            summary.add_scalar(tag='Train/Global V',
                               scalar_value=q_v,
                               global_step=num_updates)

            num_updates += 1

        if d_n['__all__']:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            episode_num_veh = len(episode_veh_step_dict)
            episode_avg_reward = episode_reward / episode_num_veh
            episode_avg_crash_veh = episode_crash_veh / episode_num_veh
            episode_avg_run_out_veh = episode_run_out / episode_num_veh
            episode_avg_arr_dest = episode_arr_dest / episode_num_veh
            if len(episode_veh_finish_step_dict) > 0:
                episode_avg_reach_step = np.mean(list(episode_veh_finish_step_dict.values()))
            else:
                episode_avg_reach_step = env.config['horizon']

            print("Total T: {} Episode Num: {} Episode T: {} Reward: {:.3f}".format(t + 1,
                                                                                    episode_num + 1,
                                                                                    episode_timesteps,
                                                                                    episode_avg_reward))

            summary.add_scalar(tag='Perf/Episode Reward',
                               scalar_value=episode_avg_reward,
                               global_step=num_updates)
            summary.add_scalar(tag='Perf/Episode Global Reward',
                               scalar_value=episode_global_reward,
                               global_step=num_updates)
            summary.add_scalar(tag='Perf/Episode Crash',
                               scalar_value=episode_avg_crash_veh,
                               global_step=num_updates)
            summary.add_scalar(tag='Perf/Episode Out',
                               scalar_value=episode_avg_run_out_veh,
                               global_step=num_updates)
            summary.add_scalar(tag='Perf/Episode Arrive',
                               scalar_value=episode_avg_arr_dest,
                               global_step=num_updates)
            summary.add_scalar(tag='Perf/Episode Length',
                               scalar_value=episode_avg_reach_step,
                               global_step=num_updates)

            # Save best model
            if episode_num >= 100 and episode_num % 5 == 0:
                eval_avg_arr_dest_list = []
                for _ in range(10):
                    obs_n, _ = env.reset()

                    eval_episode_arr_dest = 0
                    eval_episode_veh_step_dict = dict()
                    while True:
                        a_n = dict()
                        # Select according to policy
                        with torch.no_grad():
                            for agent, obs in obs_n.items():
                                a_n[agent] = (policy.select_action(np.array(obs)) +
                                              np.random.normal(0, max_action * 0.01, size=action_dim)).clip(-max_action, max_action)
                        # Perform action
                        next_obs_n, r_n, d_n, _, info = env.step(a_n)

                        for agent in list(obs_n.keys()):
                            # Remove the observations of finished agents
                            if d_n[agent]:
                                del next_obs_n[agent]

                            # Record episode metrics
                            if agent not in eval_episode_veh_step_dict:
                                eval_episode_veh_step_dict[agent] = 1
                            else:
                                eval_episode_veh_step_dict[agent] += 1
                            if info[agent]['arrive_dest']:
                                eval_episode_arr_dest += 1

                        obs_n = next_obs_n

                        if d_n['__all__']:
                            eval_avg_arr_dest_list.append(eval_episode_arr_dest / len(eval_episode_veh_step_dict))
                            break

                eval_avg_arr_dest = np.mean(eval_avg_arr_dest_list)
                if eval_avg_arr_dest > best_arr_dest or eval_avg_arr_dest > 0.85:
                    policy.save_best(models_path + "/{}_{}".format(file_name, 'best_model_{}_{:.2f}'.format(episode_num, eval_avg_arr_dest)))
                    best_arr_dest = eval_avg_arr_dest

            # Reset environment
            obs_n, _ = env.reset()

            episode_reward = 0
            episode_global_reward = 0
            episode_timesteps = 0
            episode_num += 1
            episode_crash_veh = 0
            episode_arr_dest = 0
            episode_run_out = 0
            episode_veh_step_dict = dict()
            episode_veh_finish_step_dict = dict()

        # Save model episodically
        if (t + 1) % args.save_freq == 0:
            if args.save_model:
                policy.save(models_path + "/{}".format(file_name))

    env.close()
