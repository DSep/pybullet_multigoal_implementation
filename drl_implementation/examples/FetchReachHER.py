import os
import plot
import pybullet_multigoal_gym as pmg
from pybullet_multigoal_gym.envs.task_envs.kuka_single_step_envs import \
    KukaTipOverEnv
# import gym
from drl_implementation import GoalConditionedDDPG
algo_params = {
    'hindsight': True,
    'her_sampling_strategy': 'future',
    'prioritised': False, # choose what type of HER buffer (see PrioritisedHindsightReplayBuffer)
    'memory_capacity': int(1e6),
    'actor_learning_rate': 0.001,
    'critic_learning_rate': 0.001,
    'Q_weight_decay': 0.0,
    'update_interval': 1,
    'batch_size': 128,
    'optimization_steps': 40,
    'tau': 0.05,
    'discount_factor': 0.98,
    'clip_value': 50,
    'discard_time_limit': True,
    'terminate_on_achieve': True,
    'observation_normalization': True,

    'random_action_chance': 0.2,
    'noise_deviation': 0.05,

    'training_epochs': 11,
    'training_cycles': 50,
    'training_episodes': 16,
    'testing_gap': 10,
    'testing_episodes': 30,
    'saving_gap': 1,

    # 'cuda_device_id': 0 disable cuda usage
    # 'cuda_device_full_name': 'mps'
}
seeds = [11] #, 22, 33, 44]
seed_returns = []
seed_success_rates = []
path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(path, 'Reach_HER')

for seed in seeds:

    # env = gym.make("FetchReach-v1")
    env: KukaTipOverEnv = pmg.make_env(task='tip_over',
                    gripper='parallel_jaw',
                    render=True,
                    binary_reward=False, # Switch to true for stable reward
                    joint_control=True,
                    max_episode_steps=50,
                    image_observation=False,
                    depth_image=False,
                    goal_image=False,
                    visualize_target=True,
                    camera_setup=None,
                    observation_cam_id=[0],
                    goal_cam_id=0,
                    target_range=0.3,
                    plane_position = [0.,0.,-0.5],
                    tip_penalty=-20.0,
                    force_angle_reward_factor=0.5,
                    )
    seed_path = path + '/seed'+str(seed)

    agent = GoalConditionedDDPG(algo_params=algo_params, env=env, path=seed_path, seed=seed)
    agent.run(test=False)
    # agent.run(test=True, load_network_ep=10, sleep=0.05)
    # BUG if you are loading to continue training, the data (not weights) are only from the last epoch you saved!
    # agent.run(test=False, load_network_ep=1)
    seed_returns.append(agent.statistic_dict['epoch_test_return'])
    seed_success_rates.append(agent.statistic_dict['epoch_test_success_rate'])
    del env, agent

return_statistic = plot.get_mean_and_deviation(seed_returns, save_data=True,
                                               file_name=os.path.join(path, 'return_statistic.json'))
plot.smoothed_plot_mean_deviation(path + '/returns', return_statistic, x_label='Epoch', y_label='Average returns')


success_rate_statistic = plot.get_mean_and_deviation(seed_success_rates, save_data=True,
                                                     file_name=os.path.join(path, 'success_rate_statistic.json'))
plot.smoothed_plot_mean_deviation(path + '/success_rates', success_rate_statistic,
                                  x_label='Epoch', y_label='Success rates')
