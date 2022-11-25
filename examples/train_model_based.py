import argparse
import time

import omnisafe
import torch
torch.set_num_threads(5)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--algo',
        type=str,
        default='SafeLoop',
        help='Choose from: {MBPPOLag, SafeLoop, CAP, MPCCcem',
    )
    parser.add_argument(
        '--env-id',
        type=str,
        default='SafetyPointGoal1-v0',
        help='Safexp-CarGoal1-v0,Safexp-CarGoal3-v0,HalfCheetah-v3',
    )
    #parser.add_argument('--seed', default=0, type=int, help='Define the seed of experiments')

    parser.add_argument(
        '--parallel', default=1, type=int, help='Number of cores used for calculations.'
    )
    args, unparsed_args = parser.parse_known_args()

    keys = [k[2:] for k in unparsed_args[0::2]]  # remove -- from argument
    values = [eval(v) for v in unparsed_args[1::2]]
    unparsed_dict = {k: v for k, v in zip(keys, values)}

    env = omnisafe.Env_ModelBased(args.algo,args.env_id)
    agent = omnisafe.Agent(args.algo, env=env,parallel=args.parallel, custom_cfgs=unparsed_dict)
    #agent.set_seed(args.seed)
    agent.learn()
