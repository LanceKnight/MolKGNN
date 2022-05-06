# Written by Yunchao "Lance" Liu (www.LiuYunchao.com)

from data import DataLoaderModule, get_dataset
from model import GNNModel
from monitors import LossMonitor, LossNoDropoutMonitor, \
    LogAUCMonitor, LogAUCNoDropoutMonitor, \
    PPVMonitor, PPVNoDropoutMonitor,\
    RMSEMonitor, RMSENoDropoutMonitor,\
    AccuracyMonitor, AccuracyNoDropoutMonitor,\
    F1ScoreMonitor, F1ScoreNoDropoutMonitor
from argparse import ArgumentParser
from pprint import pprint
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import os
from clearml import Task


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


    args = parser.parse_args()
    args.tot_iterations = round(len(get_dataset(
                                                dataset_name=args.dataset_name,
                                                gnn_type=gnn_type,
                                                dataset_path=args.dataset_path
                                                )['dataset'],
                                    ) * 0.8 /args.batch_size) \
                          * args.max_epochs + 1
    args.max_steps = args.tot_iterations + 1

    if use_clearml:
        task.set_name(args.task_name)
        task.add_tags(args.dataset_name)
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

    # Pretraining data module
    if enable_pretraining:
        pass  # TODO: add pretraining data module

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

        print('Creating a model from pretrained model...')
        # TODO: Load the model from the pretrained model
    else:  # if not using pretrained model
        print(f'Creating a model from scratch...')

        model = GNNModel(gnn_type, args=args)
    return model


def actual_training(model, data_module, use_clearml, gnn_type, args):
    # Add checkpoint
    monitoring_metric = 'logAUC'
    actual_training_checkpoint_dir = args.default_root_dir
    actual_training_checkpoint_callback = ModelCheckpoint(
        monitor=monitoring_metric,
        dirpath=actual_training_checkpoint_dir,
        filename=f'best_model_metric_{monitoring_metric}', #f'{data_module.dataset_name}'+'-{# epoch}-{loss}',
        save_top_k=1,
        mode='max',
        save_last=True,
        save_on_train_epoch_end=False
    )

    # Resume from the checkpoint. Temporarily disable to facilitate dubugging.
    if not args.test and not args.validate and os.path.exists(
            f'{actual_training_checkpoint_dir}/last.ckpt'):
        print('Resuming from actual training checkpoint')
        args.resume_from_checkpoint = actual_training_checkpoint_dir + \
            '/last.ckpt'

    from pytorch_lightning.callbacks import TQDMProgressBar
    prog_bar=TQDMProgressBar(refresh_rate=100)

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.callbacks=[prog_bar]
    trainer.callbacks.append(actual_training_checkpoint_callback)

    if use_clearml:
        # Loss monitors
        # trainer.callbacks.append(
        #     LossMonitor(stage='train', logger=logger, logging_interval='step'))
        trainer.callbacks.append(
            LossMonitor(stage='train', logger=logger,
                        logging_interval='epoch'))
        # trainer.callbacks.append(
        #     LossMonitor(stage='valid', logger=logger, logging_interval='step'))
        trainer.callbacks.append(
            LossMonitor(stage='valid', logger=logger,
                        logging_interval='epoch'))
        trainer.callbacks.append(
            LossNoDropoutMonitor(stage='valid', logger=logger,
                                 logging_interval='epoch'))

        # Learning rate monitors
        # trainer.callbacks.append(LearningRateMonitor(logging_interval='step'))
        trainer.callbacks.append(LearningRateMonitor(logging_interval='epoch'))

        # Other metrics monitors
        metrics = get_dataset(dataset_name=args.dataset_name,
                              gnn_type=gnn_type,
                              dataset_path=args.dataset_path
                              )['metrics']
        for metric in metrics:
            if metric == 'accuracy':
                # Accuracy monitors
                trainer.callbacks.append(
                    AccuracyMonitor(stage='train', logger=logger,
                                    logging_interval='epoch'))
                trainer.callbacks.append(
                    AccuracyMonitor(stage='valid', logger=logger,
                                    logging_interval='epoch'))
                trainer.callbacks.append(
                    AccuracyNoDropoutMonitor(stage='valid', logger=logger,
                                             logging_interval='epoch'))
                continue

            if metric == 'RMSE':
                # Accuracy monitors
                trainer.callbacks.append(
                    RMSEMonitor(stage='train', logger=logger,
                                logging_interval='epoch'))
                trainer.callbacks.append(
                    RMSEMonitor(stage='valid', logger=logger,
                                logging_interval='epoch'))
                trainer.callbacks.append(
                    RMSENoDropoutMonitor(stage='valid', logger=logger,
                                         logging_interval='epoch'))
                continue

            if metric == 'logAUC':
                # LogAUC monitors
                trainer.callbacks.append(
                    LogAUCMonitor(stage='train', logger=logger,
                                  logging_interval='epoch'))
                trainer.callbacks.append(
                    LogAUCMonitor(stage='valid', logger=logger,
                                  logging_interval='epoch'))
                trainer.callbacks.append(
                    LogAUCNoDropoutMonitor(stage='valid', logger=logger,
                                           logging_interval='epoch'))
                continue

            if metric == 'ppv':
                # PPV monitors
                trainer.callbacks.append(
                    PPVMonitor(stage='train', logger=logger, logging_interval='epoch'))
                trainer.callbacks.append(
                    PPVMonitor(stage='valid', logger=logger, logging_interval='epoch'))
                trainer.callbacks.append(
                    PPVNoDropoutMonitor(stage='valid', logger=logger,
                                        logging_interval='epoch'))
                continue

            if metric == 'f1_score':
                # F1 monitors
                trainer.callbacks.append(
                    F1ScoreMonitor(stage='train', logger=logger,
                                   logging_interval='epoch'))
                trainer.callbacks.append(
                    F1ScoreMonitor(stage='valid', logger=logger,
                                   logging_interval='epoch'))
                trainer.callbacks.append(
                    F1ScoreNoDropoutMonitor(stage='valid', logger=logger,
                                            logging_interval='epoch'))
                continue

    if args.test:
        print(f'In Testing Mode:')
        result = trainer.test(model, datamodule=data_module)
        pprint(result)
    elif args.validate:
        print(f'In Validation Mode:')
        result = trainer.validate(model, datamodule=data_module)
        pprint(result)
    else:
        print(f'In Training Mode:')
        trainer.fit(model=model, datamodule=data_module)
        print(f'In Testing Mode:')
        best_path = actual_training_checkpoint_callback.best_model_path
        print(f'best_path:{best_path}')
        model  = GNNModel.load_from_checkpoint(best_path, gnn_type=gnn_type,
                                              args=args)
        result = trainer.test(model, datamodule=data_module)


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

    # Pretrain the model if pretraining is enabled
    if enable_pretraining:
        pretraining_data_module = data_modules[1]
        # TODO: prepare the model for pretraining
        # TODO: pretrain the model

    # Prepare model for actural training
    model = prepare_actual_model(args)

    # Start actual training
    actual_training(model, actual_training_data_module, use_clearml,
                    gnn_type, args)

    if gnn_type=='kgnn':
        # Save relevant data for analyses
        model.save_atom_encoder(dir = 'utils/atom_encoder/',
        file_name='atom_encoder.pt')
        model.save_kernels(dir='utils/atom_encoder/', file_name='kernels.pt')
        model.print_graph_embedding()
        model.save_graph_embedding('utils/atom_encoder/graph_embedding')


if __name__ == '__main__':

    # The reason that gnn_type cannot be a cmd line
    # argument is that model specific arguments depends on it

    gnn_type = 'kgnn'
    # gnn_type = 'dimenet' # Not implemented
    # gnn_type = 'chironet'
    # gnn_type = 'dimenet_pp'
    # gnn_type = 'spherenet'

    use_clearml = True
    if use_clearml:
        task = Task.init(project_name=f"experiments/kgnn",
                         task_name=f"{gnn_type}",
                         tags=[],
                         reuse_last_task_id=False
                         )

        logger = task.get_logger()
        # logger = pl.loggers.tensorboard
    main(gnn_type, use_clearml)
