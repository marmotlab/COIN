import torch
import argparse
import numpy as np

from utils import set_eval_env, save_as_json
from models.coin import TD3

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3")  # Policy name (TD3)
    parser.add_argument("--env", default="intersection")  # MetaDrive environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--hidden_size", default=256, type=int, help='The number of hidden units')
    parser.add_argument("--expl_noise", default=0.01, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--model_path", default="", required=True)  # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--num_eval", default=20, type=int)  # Number of episodes for evaluation
    parser.add_argument("--num_agents", default=30, type=int) # Number of agents
    parser.add_argument("--gui", default=True, type=bool)  # Activate Metadrive gui window
    parser.add_argument("--generate_gif", default=False, type=bool) # Whether to save gifs
    args = parser.parse_args()

    print("---------------------------------------")
    print("Policy: {}, Env: {}, Seed: {}, Num Agents: {}".format(args.policy, args.env, args.seed, args.num_agents))
    print("---------------------------------------")

    env = set_eval_env(args.env, args.seed, args.num_agents)
    # Set seeds
    env.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    state_dim = 10
    obs_dim = env.observation_space['agent0'].shape[0]
    action_dim = env.action_space['agent0'].shape[0]
    max_action = float(env.action_space['agent0'].high[0])

    kwargs = {
        "obs_dim": obs_dim,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
    }

    # Initialize policy
    if args.policy == "TD3":
        # Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        kwargs['hidden_dim'] = args.hidden_size
        policy = TD3(**kwargs)
    else:
        raise NotImplementedError
    
    # load last
    # file_name = "{}_{}_{}".format(args.policy, args.env, args.seed)
    # policy_file = file_name
    # policy.load(args.model_path + "/model/" + policy_file)
    
    # load best
    policy.load_best(args.model_path + "/model")

    eval_metrics = dict(avg_arr_dest=[], avg_crash_veh=[], avg_run_out=[], avg_effi=[], avg_reach_step=[],avg_safety=[])
    for i in range(int(args.num_eval)):
        env.close()
        seed = 1000 + i * 100
        env = set_eval_env(args.env, seed, args.num_agents)
        # Reset environment
        obs_n, _ = env.reset()
        if args.env == 'intersection':
            env.render(mode="top_down", num_stack=25, camera_position=(75, 5), screen_record=True, window=args.gui)
        elif args.env == 'roundabout':
            env.render(mode="top_down", num_stack=25, camera_position=(105, 5), scaling=3.6, screen_record=True,
                        window=args.gui)
        elif args.env == 'bottleneck':
            env.render(mode='top_down', num_stack=25, camera_position=(95, 5), screen_record=True, window=args.gui)

        episode_reward = 0
        episode_timesteps = 0
        episode_crash_veh = 0
        episode_arr_dest = 0
        episode_run_out = 0
        episode_efficiency = 0
        episode_veh_step_dict = dict()
        episode_veh_finish_step_dict = dict()
        while True:
            episode_timesteps += 1
            a_n = dict()
            # Select according to policy
            for agent, obs in obs_n.items():
                a_n[agent] = (policy.select_action(np.array(obs)) +
                                np.random.normal(0, max_action * args.expl_noise,
                                                size=action_dim)).clip(-max_action, max_action)
                # a_n[agent] = (policy.select_action(np.array(obs))).clip(-max_action, max_action)

            # Perform action
            next_obs_n, r_n, d_n, _, info = env.step(a_n)
            if args.env == 'intersection':
                env.render(mode="top_down", num_stack=25, camera_position=(75, 5), screen_record=True,
                            window=args.gui)
            elif args.env == 'roundabout':
                env.render(mode="top_down", num_stack=25, camera_position=(105, 5), scaling=3.6, screen_record=True,
                            window=args.gui)
            elif args.env == 'bottleneck':
                env.render(mode='top_down', num_stack=25, camera_position=(95, 5), screen_record=True,
                            window=args.gui)

            for agent in list(obs_n.keys()):
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

            if d_n['__all__']:
                episode_num_veh = len(episode_veh_step_dict)
                episode_avg_reward = episode_reward / episode_num_veh
                episode_avg_crash_veh = episode_crash_veh / episode_num_veh
                episode_avg_run_out_veh = episode_run_out / episode_num_veh
                episode_avg_arr_des = episode_arr_dest / episode_num_veh
                episode_avg_efficiency = (episode_arr_dest - episode_crash_veh - episode_run_out) / env.config['horizon']
                episode_avg_safety = -(episode_run_out+episode_crash_veh)
                
                if len(episode_veh_finish_step_dict) > 0:
                    episode_avg_reach_step = np.mean(list(episode_veh_finish_step_dict.values()))
                else:
                    episode_avg_reach_step = env.config['horizon']
                # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
                print("Episode Num: {} Episode T: {} Success: {:.3f}".format(i,
                                                                                episode_timesteps,
                                                                                episode_avg_arr_des))

                # Save evaluation metrics
                eval_metrics['avg_arr_dest'].append(episode_avg_arr_des)
                eval_metrics['avg_crash_veh'].append(episode_avg_crash_veh)
                eval_metrics['avg_run_out'].append(episode_avg_run_out_veh)
                eval_metrics['avg_effi'].append(episode_avg_efficiency)
                eval_metrics['avg_reach_step'].append(episode_avg_reach_step)
                eval_metrics['avg_safety'].append(episode_avg_safety)

                # Save gifs
                if args.generate_gif:
                    env.top_down_renderer.generate_gif(args.model_path + '/gifs/epi{}_success{:.2f}.gif'.format(i, episode_avg_arr_des))

                break

    save_as_json(args.model_path+'/result/{}_{}_{}.json'.format(args.policy, args.num_agents, args.env), eval_metrics)
    print("Average Success Rate: {}\n".format(np.mean(eval_metrics['avg_arr_dest'])))
    env.close()


