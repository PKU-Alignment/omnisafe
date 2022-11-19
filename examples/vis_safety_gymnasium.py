import safety_gymnasium


env_name = 'SafetyPointGoal1-v0'
env = safety_gymnasium.make(env_name, render_mode='human')

obs, info = env.reset()
terminated = False

while not terminated:
    act = env.action_space.sample()
    obs, reward, cost, terminated, truncated, info = env.step(act)
    env.render()
