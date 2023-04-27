import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'examples')))

from analyze_experiment_results import my_analyze
from evaluate_saved_policy import my_evaluate
from train_from_custom_dict import train_from_dict
import subprocess


# def test_analyze_experiment_results():
#     my_analyze(path='./saved_source/test_statistics_tools',
#                parameter='algo', show_image=False)
    
# def test_evaluate_saved_policy():
#     my_evaluate(save_dir='./saved_source/PPO-{SafetyPointGoal1-v0}/seed-000-2023-03-16-12-08-52',
#                 camera_name='track', width=1, height=1, num_episodes=1)
    
# def test_plot():
#         result = subprocess.run(['python', '../examples/plot.py', '--logdir', './saved_source/test_statistics_tools/SafetyAntVelocity-v1---556c9cedab7db813a6ea3860f5921d7ccbc176d70900e709065fc2604d02b9a6/NaturalPG-{SafetyAntVelocity-v1}'], check=True)
#         print("stderr: ", result.stderr)

def test_train_from_custom_dict():
    train_from_dict(total_steps=2048, steps_per_epoch=1024)

def test_train_policy():
    result = subprocess.run(['python', '../examples/train_policy.py', '--logdir', './saved_source/test_statistics_tools/SafetyAntVelocity-v1---556c9cedab7db813a6ea3860f5921d7ccbc176d70900e709065fc2604d02b9a6/NaturalPG-{SafetyAntVelocity-v1}'], check=True)
    print("stderr: ", result.stderr)