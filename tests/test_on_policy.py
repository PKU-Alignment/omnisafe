import helpers
import omnisafe


@helpers.parametrize(
    algo=[
        'PolicyGradient',
        'PPO',
        'PPOLag',
        'NaturalPG',
        'TRPO',
        'TRPOLag',
        'PDO',
        'NPGLag',
        'CPO',
        'PCPO',
        'FOCOPS',
        'CPPOPid',
    ]
)
def test_on_policy(algo):
    env_id = 'SafetyPointGoal1-v0'
    seed = 0
    custom_cfgs = {'epochs': 1, 'steps_per_epoch': 1000, 'pi_iters': 1, 'critic_iters': 1}
    env = omnisafe.Env(env_id)
    agent = omnisafe.Agent(algo, env, custom_cfgs=custom_cfgs, parallel=1)
    agent.learn()


if __name__ == '__main__':
    test_on_policy(algo='CPO')
