import os
# from drl_implementation.examples import plot
from drl_implementation.agent.utils import plot
import pybullet_multigoal_gym as pmg
from pybullet_multigoal_gym.envs.task_envs.kuka_single_step_envs import \
    KukaTipOverEnv
# import gym
from drl_implementation import GoalConditionedDDPG
from seer.train_and_eval_configs.constants import *

from pathlib import Path
import argparse
import wandb
import importlib


def main(use_wandb, config):
    run_params = config.run_params
    env_params = config.env_params
    algo_params = config.algo_params
    rl_wandb_config = config.wandb_config
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
                name=str(Path(config.__file__).stem) + '_' + str(seed),

                # track hyperparameters and run metadata
                config=rl_wandb_config
            )          
            # # # define a metric we are interested in the minimum of
            # wandb.define_metric("actor_loss", summary="min")
            # # define a metric we are interested in the maximum of
            # wandb.define_metric("critic_loss", summary="max")
            # for i in range(10):
            #     log_dict = {
            #         "loss": random.uniform(0,1/(i+1)),
            #         "acc": random.uniform(1/(i+1),1),
            #     }
            # wandb.log(log_dict)

        env: KukaTipOverEnv = pmg.make_env(**env_params)
        seed_path = path + '/seed'+str(seed)
        if wandb.run:
            wandb.log({
                SEED: seed,
                SEEDS: seeds,
            }, step=env.total_steps)

        agent = GoalConditionedDDPG(
            algo_params=algo_params, env=env, path=seed_path, seed=seed)
        agent.run(test=run_params[IS_TEST], render=env_params['render'],
                  load_network_ep=run_params[LOAD_NETWORK_EP], sleep=run_params[SLEEP])
        # agent.run(test=True, load_network_ep=10, sleep=0.05)
        # BUG TODO Verify: if you are loading to continue training, the data (not weights) are only from the last full epoch you saved!
        # agent.run(test=False, load_network_ep=5)
        seed_returns.append(agent.statistic_dict['epoch_test_return'])
        seed_success_rates.append(
            agent.statistic_dict['epoch_test_success_rate'])

        return_statistic = plot.get_mean_and_deviation(seed_returns, save_data=True,
                                                       file_name=os.path.join(path, 'return_statistic.json'))
        if wandb.run:
            for key in return_statistic:
                wandb.log({
                    'return_statistic_' + key: return_statistic[key],
                }, step=env.total_steps)
        plot.smoothed_plot_mean_deviation(
            path + '/returns', return_statistic, x_label='Epoch', y_label='Average returns', key='Average_returns')

        success_rate_statistic = plot.get_mean_and_deviation(seed_success_rates, save_data=True,
                                                             file_name=os.path.join(path, 'success_rate_statistic.json'))
        if wandb.run:
            for key in success_rate_statistic:
                wandb.log({
                    'success_rate_statistic_' + key: success_rate_statistic[key],
                }, step=env.total_steps)
        plot.smoothed_plot_mean_deviation(path + '/success_rates', success_rate_statistic,
                                          x_label='Epoch', y_label='Success rates', key='Success_rates')
        if wandb.run:
            wandb.finish()
        del env, agent


if __name__ == '__main__':
    # Parse argument to know if we want to use wandb
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-wandb', default=False, action='store_true',
                        help='Flag to enable or disable wandb. Default: False.')
    parser.add_argument("--config", required=True, help="The complete path to the config file to use, for example seer.train_and_eval_configs.rl_eval.rl_config_eval_basic")
    parser = parser.parse_args()
    
    print("Using wandb?", parser.use_wandb)
    config = importlib.import_module(parser.config)
    main(parser.use_wandb, config)
