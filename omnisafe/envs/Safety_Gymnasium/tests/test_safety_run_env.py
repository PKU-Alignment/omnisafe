import helpers
import safety_gymnasium


@helpers.parametrize(
    agent_id=['Ant', 'Humanoid', 'Swimmer'],
    env_id=['Run'],
    version=['-v0']
)
def test_off_policy(agent_id, env_id, version):
    """test_env"""
    env_name = 'Safety' + agent_id + env_id + version
    env = safety_gymnasium.make(env_name)
    obs, _ = env.reset()
    terminled = False
    ep_ret = 0
    ep_cost = 0
    for step in range(1000):
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
