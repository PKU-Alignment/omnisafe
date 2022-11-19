import helpers
import omnisafe


@helpers.parametrize(
    algo=['PPOLag'],
    agent_id=['Point', 'Car'],
    env_id=['Goal', 'Push', 'Button'],
    level=['0', '1', '2'],
)
def test_on_policy(algo, agent_id, env_id, level):
    env_id = 'Safety' + agent_id + env_id + level + '-v0'
    # env_id = 'PointGoal1'
    custom_cfgs = {'epochs': 1, 'steps_per_epoch': 1000, 'pi_iters': 1, 'critic_iters': 1}

    env = omnisafe.Env(env_id)
    agent = omnisafe.Agent(algo, env, custom_cfgs=custom_cfgs, parallel=1)
    # agent.set_seed(seed=0)
    agent.learn()
