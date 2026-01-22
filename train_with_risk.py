import omnisafe

print("=== 使用更容易产生成本的环境 ===")

# 尝试这些环境（更容易违反约束）：
# 1. SafetyAntVelocity-v1 (速度控制，容易超速)
# 2. SafetyPointCircle1-v0 (绕圈任务，容易碰壁)
# 3. SafetyCarButton1-v0 (按钮任务，有障碍物)

env_id = 'SafetyAntVelocity-v1'  # 推荐：速度控制任务容易产生成本
print(f"环境: {env_id}")

agent = omnisafe.Agent(
    algo='PPOLag',
    env_id=env_id,
    train_terminal_cfgs={
        'parallel': 1,
        'total_steps': 5000,
        'device': 'cpu',
        'vector_env_nums': 1,
        'torch_threads': 1,
    },
    custom_cfgs={
        'algo_cfgs': {
            'steps_per_epoch': 500,
        },
        'lagrange_cfgs': {
            'cost_limit': 10.0,
        }
    }
)

print("开始训练...")
try:
    agent.learn()
    print("✅ 训练成功!")
except Exception as e:
    print(f"❌ 训练失败: {e}")
    import traceback
    traceback.print_exc()
    
    print("\n尝试更激进的环境...")
    env_id = 'SafetyPointCircle1-v0'  # 绕圈任务，更容易碰壁
    print(f"新环境: {env_id}")
    
    agent2 = omnisafe.Agent(
        algo='PPOLag',
        env_id=env_id,
        train_terminal_cfgs={
            'parallel': 1,
            'total_steps': 3000,
            'device': 'cpu',
            'vector_env_nums': 1,
            'torch_threads': 1,
        },
        custom_cfgs={
            'algo_cfgs': {
                'steps_per_epoch': 300,
            },
            'lagrange_cfgs': {
                'cost_limit': 5.0,
            }
        }
    )
    
    print("开始第二次训练...")
    agent2.learn()
