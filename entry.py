# Written by Yunchao "Lance" Liu (www.LiuYunchao.com)
from data import DataLoaderModule
import glob
from model import GNNModel
from monitors import LossMonitor, \
    LossNoDropoutMonitor,\
    LogAUC0_001to0_1Monitor,  \
    LogAUC0_001to0_1NoDropoutMonitor,\
    LogAUC0_001to1Monitor,  \
    LogAUC0_001to1NoDropoutMonitor,  \
    AUCMonitor,\
    AUCNoDropoutMonitor,\
    PPVMonitor,\
    PPVNoDropoutMonitor,\
    RMSEMonitor,\
    AccuracyMonitor,\
    F1ScoreMonitor,\
    F1ScoreNoDropoutMonitor

from argparse import ArgumentParser
from datetime import datetime
import math
from pprint import pprint
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar
import os
import os.path as osp
from clearml import Task
import time

def add_args(gnn_type):
    """
    Add arguments from three sources:
    1. default pytorch lightning arguments
    2. model specific arguments
    3. data specific arguments
    :param gnn_type: a lowercase string specifying GNN type
    :return: the arguments object
    """

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)  # default pl args
    parser = GNNModel.add_model_args(gnn_type, parser)
    parser = DataLoaderModule.add_argparse_args(parser)

    # Custom arguments
    parser.add_argument("--enable_pretraining", default=False)  # TODO: \
    parser.add_argument('--task_name', type=str, default='Unnamed')
    # Pretraining

    # Experiment labels arguments for tagging the task
    parser.add_argument("--machine", default='barium')
    parser.add_argument("--gnn_type", default=gnn_type)
    parser.add_argument("--task_comment", type=str, default='')


    args = parser.parse_args()

    if use_clearml:
        task.set_name(args.task_name)
        task.add_tags(f'model_{gnn_type}')
        task.add_tags(args.dataset_name) # args0 in scheduler
        task.add_tags(f'seed_{args.seed}') # args1
        task.add_tags(f'warm_{args.warmup_iterations}') # args2
        task.add_tags(f'epoch_{args.max_epochs}') # args3
        task.add_tags(f'peak_{args.peak_lr}') # args4
        task.add_tags(f'end_{args.end_lr}') # args5
        if gnn_type == 'kgnn':
            task.add_tags(f'layers_{args.num_layers}') # args6
            task.add_tags(f'k1_{args.num_kernel1_1hop}') # args7
            task.add_tags(f'k2_{args.num_kernel2_1hop}') # args8
            task.add_tags(f'k3_{args.num_kernel3_1hop}') # args9
            task.add_tags(f'k4_{args.num_kernel4_1hop}') # args10
            task.add_tags(f'hidden_{args.hidden_dim}') # args11
            task.add_tags(f'batch_{args.batch_size}') # args12
            task.set_comment(args.task_comment)
    with open(filename, 'a') as out_file:
        out_file.write(f'\n{args.task_comment}')

    return args


def prepare_data(args, enable_pretraining=False, gnn_type='kgnn'):
    """
    Prepare data modules for actual training, and if needed, for pretraining
    :param args: arguments for creating data modules
    :param pretraining: If True, prepare data module for pretraining as well
    :return: a list of data modules. The 0th one is always actual training data
    """

    data_modules = []

    # Actual data module
    actual_data_module = DataLoaderModule.from_argparse_args(args)
    data_modules.append(actual_data_module)

    num_train_batches = math.ceil(len(actual_data_module.dataset_train)/args.batch_size)
    num_valid_batches = math.ceil(len(actual_data_module.dataset_val) / args.batch_size)
    args.tot_iterations = (num_train_batches) * args.max_epochs + 2
    args.warmup_iterations+=2
    args.max_steps = args.tot_iterations
    args.metrics = data_modules[0].dataset['metrics']
    args.loss_func = data_modules[0].dataset['loss_func']

    print(f'entry.py::train # batches:{num_train_batches}')
    print(f'entry.py::val # batches:{num_valid_batches}')
    print(f'entry.py::args.total_iterations:{args.tot_iterations}')
    if args.train_metric:
        print(f'entry.py:: steps/epoch = num_train_batches({num_train_batches})*2 + num_valid_batches('
              f'{num_valid_batches}) = {num_train_batches*2+num_valid_batches}')
    else:
        print(f'entry.py:: steps/epoch = num_train_batches({num_train_batches}) + num_valid_batches('
              f'{num_valid_batches}) = {num_train_batches+num_valid_batches}')

    # Pretraining data module
    if enable_pretraining:
        pass  # 
        
        # TODO: add pretraining data module

    return data_modules


def prepare_actual_model(args):
    # Create actual training model using a pretrained model, if that exists
    enable_pretraining = args.enable_pretraining
    if enable_pretraining:
        # Check if pretrained model exists
        if args.pretrained_model_dir == "":
            raise Exception(
                "entry.py::pretrain_models(): pretrained_model_dir is blank")
        if not os.path.exists(args.pretrain_model_dir + '/last.ckpt'):
            raise Exception()

        print('Using pretrained model...')
        # TODO: Load the model from the pretrained model
    else:  # if not using pretrained model
        print(f'Not using pretrained model.')
        model = GNNModel(gnn_type, args=args)
    return model

def load_best_model(trainer, data_module, metric=None, args=None):
    # Load best model
    search_name = f'best*_{metric}*'
    all_files = glob.glob(osp.join(args.default_root_dir, search_name))
    if len(all_files) == 1:
        best_path = all_files[0]
    elif len(all_files) >1:
        print(f"entry::more than one best model found for {metric}!!!")
        return False
    elif len(all_files) ==0:
        print(f'No best model saved for {metric}')
        return False
    print(f"glob result:{best_path}")

    model  = GNNModel.load_from_checkpoint(best_path, gnn_type=gnn_type, args=args)
    print(f'====best_{metric}_result====:\n')
    best_result = trainer.test(model, datamodule=data_module)
    new_name = f'logs/best_{metric}_sample_scores.log'
    os.rename('logs/test_sample_scores.log', new_name)
    return best_result


def testing_procedure(trainer, data_module, args):
    print(f'In Testing Mode:')
    print(f'default_root_dir:{args.default_root_dir}')

    # Load last model
    last_path = osp.join(args.default_root_dir, 'last.ckpt')
    model  = GNNModel.load_from_checkpoint(last_path, gnn_type=gnn_type, args=args)
    print('====last_result====:\n')
    last_result = trainer.test(model, datamodule=data_module)
    os.rename('logs/test_sample_scores.log',
              'logs/last_test_sample_scores.log')

    # Save the result to a file
    filename = 'logs/test_result.log'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as out_file:
        out_file.write(f'{args.dataset_name}\n')
        out_file.write('last:\n')
        out_file.write(f'{str(last_result)}\n')

        for metric in data_module.dataset["metrics"]:
            best_result = load_best_model(trainer=trainer, data_module=data_module, metric=metric, args=args)
            if best_result is not False:
                out_file.write(f'best_{metric}:\n')
                out_file.write(f'{str(best_result)}\n')
        out_file.write(f'args:\n')
        out_file.write(f'{args}')


def actual_training(model, data_module, use_clearml, gnn_type, args):
    # Add checkpoint
    monitoring_metric = 'logAUC_0.001_0.1'
    actual_training_checkpoint_dir = args.default_root_dir
    actual_training_checkpoint_callback = ModelCheckpoint(
        monitor=monitoring_metric,
        dirpath=actual_training_checkpoint_dir,
        filename='best_model_metric_{epoch}_{logAUC_0.001_0.1}',
        save_top_k=1,
        mode='max',
        save_last=True,
        save_on_train_epoch_end=False
    )

    best_AUC_callback = ModelCheckpoint(
        monitor='AUC',
        dirpath=actual_training_checkpoint_dir,
        filename='best_model_metric_{epoch}_{AUC}',
        save_top_k=1,
        mode='max',
        save_last=True,
        save_on_train_epoch_end=False
    )

    best_AUC_0_001_0_1_callback = ModelCheckpoint(
        monitor='logAUC_0.001_0.1',
        dirpath=actual_training_checkpoint_dir,
        filename='best_model_metric_{epoch}_{logAUC_0.001_1}',
        save_top_k=1,
        mode='max',
        save_last=True,
        save_on_train_epoch_end=False
    )

    best_loss_callback = ModelCheckpoint(
        monitor='loss',
        dirpath=actual_training_checkpoint_dir,
        filename='best_model_metric_{epoch}_{loss}',
        save_top_k=1,
        mode='min',
        save_last=True,
        save_on_train_epoch_end=False
    )


    prog_bar=TQDMProgressBar(refresh_rate=500)
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.callbacks=[prog_bar]
    trainer.callbacks.append(actual_training_checkpoint_callback)

    trainer.callbacks.append(best_AUC_callback)
    trainer.callbacks.append(best_AUC_0_001_0_1_callback)
    trainer.callbacks.append(best_loss_callback)

    if use_clearml:
        trainer.callbacks.append(LossMonitor(stage='train', logger=logger, logging_interval='epoch'))
        trainer.callbacks.append(LossMonitor(stage='valid', logger=logger, logging_interval='epoch'))
        if args.train_metric:
            trainer.callbacks.append(LossNoDropoutMonitor(stage='valid', logger=logger, logging_interval='epoch'))

        # Learning rate monitors
        trainer.callbacks.append(LearningRateMonitor(logging_interval='step'))

        # Other metrics monitors
        metrics = data_module.dataset['metrics']
        for metric in metrics:
            if metric == 'accuracy':
                trainer.callbacks.append(AccuracyMonitor(stage='valid', logger=logger, logging_interval='epoch'))
                continue

            if metric == 'RMSE':
                trainer.callbacks.append(RMSEMonitor(stage='valid', logger=logger, logging_interval='epoch'))
                continue

            if metric == 'logAUC_0.001_0.1':
                trainer.callbacks.append(LogAUC0_001to0_1Monitor(stage='valid', logger=logger, logging_interval='epoch'))
                if args.train_metric:
                    trainer.callbacks.append(
                        LogAUC0_001to0_1NoDropoutMonitor(stage='valid', logger=logger, logging_interval='epoch')
                    )
                continue

            if metric == 'logAUC_0.001_1':
                trainer.callbacks.append(LogAUC0_001to1Monitor(stage='valid', logger=logger, logging_interval='epoch'))
                if args.train_metric:
                    trainer.callbacks.append(
                        LogAUC0_001to1NoDropoutMonitor(stage='valid', logger=logger, logging_interval='epoch')
                    )
                continue

            if metric == 'AUC':
                trainer.callbacks.append(AUCMonitor(stage='valid', logger=logger, logging_interval='epoch'))
                if args.train_metric:
                    trainer.callbacks.append(
                        AUCNoDropoutMonitor(stage='valid', logger=logger, logging_interval='epoch')
                    )
                continue

            if metric == 'ppv':
                trainer.callbacks.append(PPVMonitor(stage='valid', logger=logger, logging_interval='epoch'))
                if args.train_metric:
                    trainer.callbacks.append(
                        PPVNoDropoutMonitor(stage='valid', logger=logger, logging_interval='epoch')
                    )
                continue

            if metric == 'f1_score':
                trainer.callbacks.append(F1ScoreMonitor(stage='valid', logger=logger, logging_interval='epoch'))
                if args.train_metric:
                    trainer.callbacks.append(
                        F1ScoreNoDropoutMonitor(stage='valid', logger=logger, logging_interval='epoch')
                    )
                continue

    if args.test:
        testing_procedure(trainer, data_module, args)
    elif args.validate:
        print(f'In Validation Mode:')
        result = trainer.validate(model, datamodule=data_module)
        pprint(result)
    else:
        print(f'In Training Mode:')
        trainer.fit(model=model, datamodule=data_module)
        
        # In testing Mode
        testing_procedure(trainer, data_module, args)
        if gnn_type=='kgnn':
            model.save_kernels(dir='analyses/atom_encoder/', file_name='kernels.pt')
            model.print_graph_embedding()
            model.save_graph_embedding('analyses/atom_encoder/graph_embedding')


def main(gnn_type, use_clearml):
    """
    the main process that defines model and data
    also trains and evaluate the model
    :param gnn_type: the GNN used for prediction
    :param logger: ClearML for logging the metric
    :return: None
    """

    # Get arguments
    args = add_args(gnn_type)

    # Set seed
    pl.seed_everything(args.seed)

    # Prepare data
    enable_pretraining = args.enable_pretraining
    print(f'enable_pretraining:{enable_pretraining}')
    args.gnn_type = gnn_type
    data_modules = prepare_data(args, enable_pretraining) # A list of

    # data_module to accommodate different pretraining data
    actual_training_data_module = data_modules[0]


    # Prepare model for actural training
    model = prepare_actual_model(args)

    # Start actual training
    actual_training(model, actual_training_data_module, use_clearml,
                    gnn_type, args)


if __name__ == '__main__':
    start_sys_time = datetime.now()
    print(f'scheduler start time:{start_sys_time}')
    start = time.time()
    Task.set_offline(offline_mode=True)
    # The reason that gnn_type cannot be a cmd line
    # argument is that model specific arguments depends on it
    gnn_type = 'kgnn'
    # gnn_type = 'chironet'
    # gnn_type = 'dimenet_pp'
    # gnn_type = 'spherenet'
    # gnn_type = 'schnet'


    print(f'========================')
    print(f'Runing model: {gnn_type}')
    print(f'========================')


    filename = 'logs/task_info.log'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as out_file:
        use_clearml = False
        if use_clearml:
            task = Task.init(project_name=f"HyperParams/kgnn",
                             task_name=f"{gnn_type}",
                             tags=[],
                             reuse_last_task_id=False
                             )
            out_file.write(f'task_id:{task.id}')
            out_file.write('\n')

            logger = task.get_logger()
        main(gnn_type, use_clearml)
        end = time.time()
        run_time = end-start
        run_time_str = f'run_time:{math.floor(run_time/3600)}h{math.floor((run_time)%3600/60)}m' \
                       f'{math.floor(run_time%60)}s'
        print(run_time_str)
        end_sys_time = datetime.now()
        print(f'task finsh time:{end_sys_time}')
        out_file.write(run_time_str)
        out_file.write(f'task start time:{start_sys_time}')
        out_file.write(f'task finsh time:{end_sys_time}')

    print(f'========================')
    print(f'Runing model: {gnn_type}')
    print(f'========================')
