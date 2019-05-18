import argh
import os
import subprocess
import itertools
import random

# models = ['SHL', 'CNN']
models = ['SHL']

# datasets = ['mnist --transform pad', 'mnist_bg_rot --transform pad', 'cifar10mono'] #, 'norb --transform pad']
# datasets = ['mnist_noise_1 --transform pad', 'mnist_bg_rot --transform pad', 'cifar10mono']
datasets = ['mnist_bg_rot --transform pad', 'cifar10mono']
# datasets = ['mnist_noise_1 --transform pad']

lrds = ['1.0']
wds = ['0.0']
# lrs = ['5e-3', '1e-2', '2e-2', '5e-2', '1e-1', '2e-1', '5e-1']
lrs = ['1e-2', '2e-2', '5e-2', '1e-1']
# lrs = ['2e-1', '1e-1', '5e-2', '2e-2', '1e-2', '5e-3']

batch_size = 50

# flagss = ['', '--real']
# flagss = ['', '--real', '--fixed-perm']
# flagss = ['--fixed-perm', '--fixed-perm --real']
# flagss = ['--fixed-perm --depth 1']
perms = ['i', 'b']
depths = ['1', '2', '3']
reals = ['--real']
ortho_inits = ['', '--ortho-init']

# trials = list(range(1))
trials = [0,1,2]

# ResNet experiments
# models = ['ResNet']
# datasets = ['cifar10_']
# lrs = ['1e-1', '2e-1']
# lrds = ['0.1', '0.2']
# wds = ['0.0', '1e-4']
# # wds = ['2e-4', '5e-4']
# flagss = ['', '--last-layer fc', '--last-layer b']
# trials = [2]
# batch_size = 256


epochs = [100]
epoch = 100


# seeds = [1,2]


@argh.arg("run_name", help="Directory to store the run; will be created if necessary")
@argh.arg("--machines", help="Total number of machines to use")
# @argh.arg('-d', "--datasets", nargs='+', type=str, help = "Datasets")
def run(run_name, machines=1):
    os.makedirs(run_name, exist_ok=True)

    params = []
    outputs = []

    commit_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip().decode("utf-8")
    stuff = itertools.product(trials, models, datasets, lrs, lrds, wds, perms, depths, reals, ortho_inits)
    hparams = list(stuff)
    random.shuffle(hparams)
    for trial, model, dataset, lr, lrd, wd, perm, depth, real, ortho_init in hparams:
	# python main.py --dataset cifar10mono --result-dir butterfly --lr 1e-2 --epochs 100 model SHL --class-type b
        param = [
            '--dataset', dataset,
            '--result-dir', run_name,
            # '--name', model,
            '--lr', lr,
            '--lr-decay', lrd,
            '--weight-decay', wd,
            '--trial-id', str(trial),
            '--epochs', str(epoch),
            '--batch-size', str(batch_size),
            # '--trials', '3',
            'model', model,
            '--class-type', 'b',
            '--perm', perm,
            '--depth', depth,
            real, ortho_init
            # flags
            ]
        params.append(" ".join(param))
        # outputs.append(output)

    cmds = []
    for i in range(machines):
        header = " ".join(['python main.py'])
        cmds = [f'{header} {p}' for p in params[i::machines]]
        with open(f"{run_name}/cmds{i}.sh", "w") as cmd_log:
            cmd_log.writelines('\n'.join(cmds))
        # with open(f'{run_name}/eval{i}.sh', 'w') as eval_cmd:
        #     cmd = f'python ../python/ana.py -p 64 -s -o {run_name} ' + ' '.join(outputs[i::machines])
        #     eval_cmd.write(cmd + '\n')
        #     cmd = f'python ../python/sim.py -o {run_name} ' + ' '.join(outputs[i::machines])
        #     eval_cmd.write(cmd + '\n')


    # all_cmds = [f'"{cmd0} {p}"' for p in params[0::2]] \
    # + [f'"{cmd1} {p}"' for p in params[1::2]]
    # parallel_cmd = " ".join(['parallel',
    #         ':::',
    #         *all_cmds
    #         ])
    # print(parallel_cmd)
    # with open(f"{run_name}/cmds.sh", "w") as cmd_log:
    #     cmd_log.writelines('\n'.join(all_cmds))
    # subprocess.run(parallel_cmd, shell=True)



if __name__ == '__main__':
    _parser = argh.ArghParser()
    _parser.set_default_command(run)
    _parser.dispatch()
