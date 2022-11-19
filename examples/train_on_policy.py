import argparse
import time

import omnisafe


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--algo',
        type=str,
        default='PPO',
        help='Choose from: {PolicyGradient, PPO, PPOLag, NaturalPG, TRPO, TRPOLag, PDO, NPGLag, CPO, PCPO, FOCOPS, CPPOPid',
    )
    parser.add_argument(
        '--env-id',
        type=str,
        default='SafetyPointGoal1-v0',
        help='The name of test environment',
    )
    parser.add_argument(
        '--parallel', default=1, type=int, help='Number of paralleled progress for calculations.'
    )
    args, unparsed_args = parser.parse_known_args()

    keys = [k[2:] for k in unparsed_args[0::2]]  # remove -- from argument
    values = [eval(v) for v in unparsed_args[1::2]]
    unparsed_dict = {k: v for k, v in zip(keys, values)}

    env = omnisafe.Env(args.env_id)
    agent = omnisafe.Agent(args.algo, env, parallel=args.parallel, custom_cfgs=unparsed_dict)
    agent.learn()
