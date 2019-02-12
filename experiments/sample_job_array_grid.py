from spinup.utils.run_utils import ExperimentGrid
from spinup.algos.sac_pytorch.sac_pytorch import sac_pytorch
import time

if __name__ == '__main__':
    import argparse
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--num_runs', type=int, default=3)
    parser.add_argument('--seed', type=int ,default=0)
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    args = parser.parse_args()

    eg = ExperimentGrid(name='sac-baseline')
    eg.add('env_name', args.env, '', True)
    eg.add('seed', [args.seed])
    eg.add('epochs', 200)
    eg.add('steps_per_epoch', 5000)
    eg.run(sac_pytorch, num_cpu=args.cpu)

    print('\n###################################### GRID EXP END ######################################')
    print('total time for grid experiment:',time.time()-start_time)
