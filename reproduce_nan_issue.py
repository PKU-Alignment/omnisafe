"""
Reproduce NaN cost assertion in PPOLag with zero cost_limit.
Expected error: AssertionError: cost for updating lagrange multiplier is nan
"""

import omnisafe

def reproduce_nan_bug():
    """Reproduce the NaN assertion bug in PPOLag."""
    
    print("=" * 60)
    print("Reproducing PPOLag NaN assertion bug")
    print("=" * 60)

    agent = omnisafe.Agent(
        algo='PPOLag',
        env_id='SafetyPointGoal1-v0',  # A relatively safe environment
        train_terminal_cfgs={
            'total_steps': 2000,      # Enough to trigger within first epoch
            'parallel': 1,
            'device': 'cpu',
        },
        custom_cfgs={
            'algo_cfgs': {
                'steps_per_epoch': 200,
            },
            'lagrange_cfgs': {
                'cost_limit': 0.0     # This is the key trigger
            }
        }
    )
    
    print("Configuration:")
    print(f"  Algorithm: PPOLag")
    print(f"  Environment: SafetyPointGoal1-v0")
    print(f"  Cost limit: 0.0")
    print()
    print("Expected behavior: Will crash with AssertionError")
    print("Actual behavior: ...")
    print()
    
    agent.learn()

if __name__ == '__main__':
    reproduce_nan_bug()