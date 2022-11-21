import argparse

import safety_gymnasium


def run_random(env_name):
    env = safety_gymnasium.make(env_name, render_mode='rgb_array')
    # env.seed(0)
    obs, _ = env.reset()
    terminled = False
    ep_ret = 0
    ep_cost = 0
    while True:
        if terminled:
            print('Episode Return: %.3f \t Episode Cost: %.3f' % (ep_ret, ep_cost))
            ep_ret, ep_cost = 0, 0
            obs, _ = env.reset()
        assert env.observation_space.contains(obs)
        act = env.action_space.sample()
        assert env.action_space.contains(act)
        # Use the environment's built_in max_episode_steps
        if hasattr(env, '_max_episode_steps'):
            max_ep_len = env._max_episode_steps

        obs, reward, cost, terminled, truncated, info = env.step(act)

        ep_ret += reward
        ep_cost += cost


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='SafetyPointGoal0-v0')
    args = parser.parse_args()
    run_random(args.env)
