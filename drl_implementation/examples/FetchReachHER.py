import os
# from drl_implementation.examples import plot
from drl_implementation.agent.utils import plot
import pybullet_multigoal_gym as pmg
from pybullet_multigoal_gym.envs.task_envs.kuka_single_step_envs import \
    KukaTipOverEnv
# import gym
from drl_implementation import GoalConditionedDDPG
from seer.evaluation_tools.constants import *
import seer.evaluation_tools.rl_config_eval_basic
from seer.evaluation_tools.rl_config_eval_basic import run_params, env_params, algo_params
from seer.evaluation_tools.rl_config_eval_basic import wandb_config as rl_wandb_config
import argparse
import wandb


def main(use_wandb=True):
    seeds = [11]
    seed_returns = []
    seed_success_rates = []
    path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(path, run_params[PATH])

    for seed in seeds:
        if use_wandb:
            # start a new wandb run to track this script
            wandb.init(
                # set the wandb project where this run will be logged
                project='seer',
                entity='general-team',
                reinit=True,
                # name=os.path.realpath(rl_config_eval_basic.__file__) + '_' + str(seed),
                
                # track hyperparameters and run metadata
                config=rl_wandb_config
            )
        if wandb.run:
            wandb.log({
                SEED: seed,
                SEEDS: seeds,
            })

        env: KukaTipOverEnv = pmg.make_env(**env_params)
        seed_path = path + '/seed'+str(seed)

        agent = GoalConditionedDDPG(algo_params=algo_params, env=env, path=seed_path, seed=seed)
        agent.run(test=run_params[IS_TEST], render=env_params['render'],
                  load_network_ep=run_params[LOAD_NETWORK_EP], sleep=run_params[SLEEP])
        # agent.run(test=True, load_network_ep=10, sleep=0.05)
        # BUG TODO Verify: if you are loading to continue training, the data (not weights) are only from the last full epoch you saved!
        # agent.run(test=False, load_network_ep=5)
        seed_returns.append(agent.statistic_dict['epoch_test_return'])
        seed_success_rates.append(agent.statistic_dict['epoch_test_success_rate'])
        del env, agent

        return_statistic = plot.get_mean_and_deviation(seed_returns, save_data=True,
                                                    file_name=os.path.join(path, 'return_statistic.json'))
        if wandb.run:
            for key in return_statistic:
                wandb.log({
                    'return_statistic_' + key: return_statistic[key],
                })
        plot.smoothed_plot_mean_deviation(path + '/returns', return_statistic, x_label='Epoch', y_label='Average returns', key='Average_returns')


        success_rate_statistic = plot.get_mean_and_deviation(seed_success_rates, save_data=True,
                                                            file_name=os.path.join(path, 'success_rate_statistic.json'))
        if wandb.run:
            for key in success_rate_statistic:
                wandb.log({
                    'success_rate_statistic_' + key: success_rate_statistic[key],
                })
        plot.smoothed_plot_mean_deviation(path + '/success_rates', success_rate_statistic,
                                        x_label='Epoch', y_label='Success rates', key='Success_rates')
        if wandb.run:
            wandb.finish()


if __name__ == '__main__':
    # Parse argument to know if we want to use wandb
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_wandb', action='store_true')
    parser = parser.parse_args()
    main(parser.use_wandb)
