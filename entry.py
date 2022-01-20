# Written by Yunchao "Lance" Liu (www.LiuYunchao.com)
# Adapted from Graphormer (https://github.com/microsoft/Graphormer)

from data import DataLoaderModule, get_dataset
from model import GNNModel
from monitors import LossMonitor, LossNoDropoutMonitor, LogAUCMonitor, \
    LogAUCNoDropoutMonitor, PPVMonitor, PPVNoDropoutMonitor, \
    AccuracyMonitor, AccuracyNoDropoutMonitor

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
    print(f'entry.py::parser:{parser} GNNModel:{GNNModel}')
    parser = DataLoaderModule.add_argparse_args(parser)

    # Custom arguments
    parser.add_argument("--enable_pretraining", default=False)  # TODO: \
    # Pretraining

    args = parser.parse_args()
    args.max_steps = args.tot_iterations + 1
    print(args)
    return args


def prepare_data(args, enable_pretraining=False):
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
        if args.pretrained_model_dir is "":
            raise Exception(
                "entry.py::pretrain_models(): pretrained_model_dir is blank")
        if not os.path.exists(args.pretrain_model_dir + '/last.ckpt'):
            raise Exception()

        print('Creating a model from pretrained model...')
        # TODO: Load the model from the pretrained model
    else:  # if not using pretrained model
        print(f'Creating a model from scratch...')

        model = GNNModel(gnn_type, args.num_layers, args.input_dim,
                         args.hidden_dim, args.output_dim,
                         args.warmup_iterations, args.tot_iterations,
                         args.peak_lr, args.end_lr)
    return model


def actual_training(model, data_module, args):
    # Add checkpoint
    actual_training_checkpoint_dir = args.default_root_dir
    actual_training_checkpoint_callback = ModelCheckpoint(
        dirpath=actual_training_checkpoint_dir,
        filename=data_module.dataset_name,
        save_last=True
    )

    # Resume from the checkpoint
    print(f'dir:{actual_training_checkpoint_dir}')
    if not args.test and not args.validate and os.path.exists(
            f'{actual_training_checkpoint_dir}/last.ckpt'):
        print('Resuming from actual training checkpoint')
        args.resume_from_checkpoint = actual_training_checkpoint_dir + \
                                      '/last.ckpt'

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.callbacks.append(actual_training_checkpoint_callback)
    print(f'max_epoch:{trainer.max_epochs}')


    # Loss monitors
    trainer.callbacks.append(
        LossMonitor(stage='train', logger=logger, logging_interval='step'))
    trainer.callbacks.append(
        LossMonitor(stage='train', logger=logger,
                    logging_interval='epoch'))
    trainer.callbacks.append(
        LossMonitor(stage='valid', logger=logger, logging_interval='step'))
    trainer.callbacks.append(
        LossMonitor(stage='valid', logger=logger,
                    logging_interval='epoch'))
    trainer.callbacks.append(
        LossNoDropoutMonitor(stage='valid', logger=logger,
                             logging_interval='epoch'))
    #
    # # LogAUC monitors
    # trainer.callbacks.append(
    #     LogAUCMonitor(stage='train', logger=logger, logging_interval='epoch'))
    # trainer.callbacks.append(
    #     LogAUCMonitor(stage='valid', logger=logger, logging_interval='epoch'))
    # trainer.callbacks.append(
    #     LogAUCNoDropoutMonitor(stage='valid', logger=logger,
    #                            logging_interval='epoch'))
    #
    # # PPV monitors
    # trainer.callbacks.append(
    #     PPVMonitor(stage='train', logger=logger, logging_interval='epoch'))
    # trainer.callbacks.append(
    #     PPVMonitor(stage='valid', logger=logger, logging_interval='epoch'))
    # trainer.callbacks.append(
    #     PPVNoDropoutMonitor(stage='valid', logger=logger,
    #                         logging_interval='epoch'))

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


    # Learning rate monitors
    # trainer.callbacks.append(LearningRateMonitor(logging_interval='step'))
    trainer.callbacks.append(LearningRateMonitor(logging_interval='epoch'))

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


def main(gnn_type):
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
    data_modules = prepare_data(args, enable_pretraining)
    actual_training_data_module = data_modules[0]

    # Pretrain the model if pretraining is enabled
    if enable_pretraining:
        pretraining_data_module = data_modules[1]
        # TODO: prepare the model for pretraining
        # TODO: pretrain the model

    # Prepare model for actural training
    model = prepare_actual_model(args)

    # Start actual training
    actual_training(model, actual_training_data_module, args)


if __name__ == '__main__':
    gnn_type = 'kgnn'  # The reason that gnn_type cannot be a cmd line
    # argument is that model specific arguments depends on it

    task = Task.init(project_name=f"Tests/{gnn_type}",
                     task_name="improving_performance",
                     tags=[gnn_type, "debug"])
    logger = task.get_logger()
    main(gnn_type)
