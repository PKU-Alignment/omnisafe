import omnisafe


env = omnisafe.Env('SafetyPointGoal1-v0')

custom_dict = {'epochs': 1, 'data_dir': './runs'}
agent = omnisafe.Agent('PPOLag', env, custom_cfgs=custom_dict)
agent.learn()

# obs = env.reset()
# for i in range(1000):
#     action, _states = agent.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     env.render()
#     if done:
#         obs = env.reset()
# env.close()
