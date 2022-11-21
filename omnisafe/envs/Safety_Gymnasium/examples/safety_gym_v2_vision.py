import argparse
import os

# import gymnasium
import safety_gymnasium
from gymnasium.utils.save_video import save_video


WORKDIR = os.path.abspath('.')
DIR = os.path.join(WORKDIR, 'omnisafe/envs/Safety_Gymnasium/examples', 'cached_test_vision_video')


def run_random(env_name):
    env = safety_gymnasium.make(env_name)
    # env.seed(0)
    obs, _ = env.reset()
    terminled = False
    ep_ret = 0
    ep_cost = 0
    render_list = []
    for i in range(1001):
        if terminled:
            print('Episode Return: %.3f \t Episode Cost: %.3f' % (ep_ret, ep_cost))
            ep_ret, ep_cost = 0, 0
            obs, _ = env.reset()
            save_video(
                frames=render_list,
                video_folder=DIR,
                name_prefix=f'test_vision_output',
                fps=30,
            )
            render_list = []
        assert env.observation_space.contains(obs)
        act = env.action_space.sample()
        assert env.action_space.contains(act)
        # Use the environment's built_in max_episode_steps
        if hasattr(env, '_max_episode_steps'):
            max_ep_len = env._max_episode_steps
        render_list.append(obs['vision'])
        obs, reward, cost, terminled, truncated, info = env.step(act)

        ep_ret += reward
        ep_cost += cost


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='SafetyCarGoal0Vision-v0')
    args = parser.parse_args()
    run_random(args.env)
