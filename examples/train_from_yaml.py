import omnisafe


env = omnisafe.Env('SafetyPointGoal1-v0')

agent = omnisafe.Agent('PPOLag', env)
agent.learn()

# obs = env.reset()
# for i in range(1000):
#     action, _states = agent.predict(obs, deterministic=True)
#     obs, reward, cost, done, info = env.step(action)
#     env.render()
#     if done:
#         obs = env.reset()
# env.close()
