from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class MetricMonitor(Callback):
    """
    A base class for all metric monitors. This callback is used by a pytorch
    lightning trainer. It reports metrics to the clearml logger.
    """

    def __init__(self, stage='train', metric=None, logger=None,
                 logging_interval=None, title=None, series=None):
        """
        :param stage: a string. The values can only be "both", "train",
        or "valid"
        :param metric:
        :param logger:
        :param logging_interval:
        :param title:
        :param series: the legend used for the clearml polot. The default is
        to use the smae string as the stage. However, there are cases when
        we want to have a differnt label than the stage, such as "train" and
        "train_no_dropout"
        """
        if logging_interval not in (None, "step", "epoch"):
            raise MisconfigurationException(
                "monitors.py::MetricMonitor: logging_interval should be "
                "`step` or `epoch` or `None`.")
        if metric is None:
            raise MisconfigurationException(
                "monitors.py::MetricMonitor: metric is not specified")
        if stage not in ('both', 'train', 'valid'):
            raise MisconfigurationException(
                f"monitors.py::MetricMonitor: input 'stage' argument = "
                f"{stage}, which cannot be recognized")
        self.logger = logger
        self.metric = metric
        self.logging_interval = logging_interval
        self.stage = stage
        self.title = title
        self.series = series

    def on_train_batch_end(self, trainer, pl_module, outputs, batch=None,
                 batch_idx=None, dataloader_idx=None):
        """
        Report metrics in each iteration
        :param trainer: pytorch lightning trainer
        :param pl_module: model used, a LightningModule class
        :param outputs: outputs from each iteration. It is from the return
        of training_step() function defined in the model
        :param batch: Not used but required as a inheried class of Callback
        :param batch_idx: Not used but required as a inheried class of Callback
        :param dataloader_idx: Not used but required as a inheried class of Callback
        :return: None
        """
        if 'train' in self.stage:
            if self.logging_interval == "step":
                series = self.series if self.series is not None else 'train'
                # print(f'monitors.py::on_train_batch_end:title={self.title}, '
                #       f'series={series}, value={outputs[self.metric]}, '
                #       f'trainer.global_step={trainer.global_step}')
                self.logger.report_scalar(title=self.title, series=series,
                                          value=outputs[self.metric],
                                          iteration=trainer.global_step)

    def on_train_epoch_end(self, trainer, pl_module):
        if 'train' in self.stage:
            if self.logging_interval == "epoch":
                outputs = pl_module.train_epoch_outputs
                series = self.series if self.series is not None else 'train'
                # print(
                #     f'on_train_epoch_end: title={self.title}, series='
                #     f'{series}, value={outputs[self.metric]}, '
                #     f'trainer.current_epoch={trainer.current_epoch}')
                self.logger.report_scalar(title=self.title, series=series,
                                          value=outputs[self.metric],
                                          iteration=trainer.current_epoch)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch,
                                batch_idx, unused=0):
        if 'valid' in self.stage:
            if self.logging_interval == "step":
                series = self.series if self.series is not None else 'valid'
                # print(
                #     f'on_validation_batch_end title={self.title}, series='
                #     f'{series}, value={outputs[self.metric]}, '
                #     f'trainer.global_step={trainer.global_step}')
                self.logger.report_scalar(title=self.title, series=series,
                                          value=outputs[self.metric],
                                          iteration=trainer.global_step)

    def on_validation_epoch_end(self, trainer, pl_module):
        if 'valid' in self.stage:
            if self.logging_interval == "epoch":
                outputs = pl_module.valid_epoch_outputs
                series = self.series if self.series is not None else 'valid'
                # print(
                #     f'on_validation_epoch_end title={self.title}, series='
                #     f'{series}, value={outputs[self.metric]}, '
                #     f'trainer.current_epoch={trainer.current_epoch}')
                self.logger.report_scalar(title=self.title, series=series,
                                          value=outputs[self.metric],
                                          iteration=trainer.current_epoch)


# Loss Monitors
class LossMonitor(MetricMonitor):
    def __init__(self, stage='train', logger=None, logging_interval=None,
                 title=None):
        super(LossMonitor, self).__init__(stage=stage, metric="loss",
                                          logger=logger,
                                          logging_interval=logging_interval,
                                          title=f'loss_by_{logging_interval}')


class LossNoDropoutMonitor(MetricMonitor):
    def __init__(self, stage='train', logger=None, logging_interval=None,
                 title=None):
        super(LossNoDropoutMonitor, self).__init__(stage=stage,
                                                   metric="loss_no_dropout",
                                                   logger=logger,
                                                   logging_interval=logging_interval,
                                                   title=f'loss_by_'
                                                         f'{logging_interval}',
                                                   series='no_dropout')


# LogAUC0.001_0.1 Monitors
class LogAUC0_001to0_1Monitor(MetricMonitor):
    def __init__(self, stage='valid', logger=None, logging_interval=None,
                 title=None):
        super(LogAUC0_001to0_1Monitor, self).__init__(stage=stage,
                                                      metric="logAUC_0.001_0.1",
                                                      logger=logger,
                                                      logging_interval=logging_interval,
                                                      title=f'logAUC_by_{logging_interval}')

# LogAUC_0.001_1 Monitors
class LogAUC0_001to1Monitor(MetricMonitor):
    def __init__(self, stage='valid', logger=None, logging_interval=None,
                 title=None):
        super(LogAUC0_001to1Monitor, self).__init__(stage=stage,
                                                      metric="logAUC_0.001_1",
                                                      logger=logger,
                                                      logging_interval=logging_interval,
                                                      title=f'logAUC_by_{logging_interval}')


class LogAUC0_001to1NoDropoutMonitor(MetricMonitor):
    def __init__(self, stage='valid', logger=None, logging_interval=None,
                 title=None):
        super(LogAUC0_001to1NoDropoutMonitor, self).__init__(stage=stage,
                                                     metric="logAUC_no_dropout",
                                                     logger=logger,
                                                     logging_interval=logging_interval,
                                                     title=f'logA'
                                                           f'UC_by_'
                                                           f'{logging_interval}',
                                                     series='no_dropout')

# AUC
class AUCMonitor(MetricMonitor):
    def __init__(self, stage='valid', logger=None, logging_interval=None,
                 title=None):
        super(AUCMonitor, self).__init__(stage=stage,
                                                      metric="AUC",
                                                      logger=logger,
                                                      logging_interval=logging_interval,
                                                      title=f'AUC_by_{logging_interval}')

class AUCNoDropoutMonitor(MetricMonitor):
    def __init__(self, stage='valid', logger=None, logging_interval=None,
                 title=None):
        super(AUCNoDropoutMonitor, self).__init__(stage=stage,
                                                     metric="AUC_no_dropout",
                                                     logger=logger,
                                                     logging_interval=logging_interval,
                                                     title=f'AUC_by_{logging_interval}',
                                                     series='no_dropout')

# PPV Monitors
class PPVMonitor(MetricMonitor):
    def __init__(self, stage='valid', logger=None, logging_interval=None,
                 title=None):
        super(PPVMonitor, self).__init__(stage=stage, metric="ppv",
                                         logger=logger,
                                         logging_interval=logging_interval,
                                         title=f'PPV_by_{logging_interval}')


class PPVNoDropoutMonitor(MetricMonitor):
    def __init__(self, stage='valid', logger=None, logging_interval=None,
                 title=None):
        super(PPVNoDropoutMonitor, self).__init__(stage=stage,
                                                  metric="ppv_no_dropout",
                                                  logger=logger,
                                                  logging_interval=logging_interval,
                                                  title=f'PPV_by_{logging_interval}',
                                                  series='no_dropout')

class AccuracyMonitor(MetricMonitor):
    def __init__(self, stage='valid', logger =None, logging_interval=None,
                 title = None ):
        super(AccuracyMonitor, self).__init__(stage=stage,
                                                  metric="accuracy",
                                                  logger=logger,
                                                  logging_interval=logging_interval,
                                                  title=f'accuracy_by_'
                                                        f'{logging_interval}')


class AccuracyNoDropoutMonitor(MetricMonitor):
    def __init__(self, stage='valid', logger =None, logging_interval=None,
                 title = None ):
        super(AccuracyNoDropoutMonitor, self).__init__(stage=stage,
                                                  metric="accuracy_no_dropout",
                                                  logger=logger,
                                                  logging_interval=logging_interval,
                                                  title=f'accuracy_by_'
                                                        f'{logging_interval}',
                                              series='no_dropout')

# RMSE
class RMSEMonitor(MetricMonitor):
    def __init__(self, stage='valid', logger =None, logging_interval=None,
                 title = None ):
        super(RMSEMonitor, self).__init__(stage=stage,
                                                  metric="RMSE",
                                                  logger=logger,
                                                  logging_interval=logging_interval,
                                                  title=f'RMSE_by_'
                                                        f'{logging_interval}')


class RMSENoDropoutMonitor(MetricMonitor):
    def __init__(self, stage='valid', logger =None, logging_interval=None,
                 title = None ):
        super(RMSENoDropoutMonitor, self).__init__(stage=stage,
                                                  metric="RMSE_no_dropout",
                                                  logger=logger,
                                                  logging_interval=logging_interval,
                                                  title=f'RMSE_by_'
                                                        f'{logging_interval}',
                                              series='no_dropout')

        
# F1 score
class F1ScoreMonitor(MetricMonitor):
    def __init__(self, stage='valid', logger =None, logging_interval=None,
                 title = None ):
        super(F1ScoreMonitor, self).__init__(stage=stage,
                                                  metric="f1_score",
                                                  logger=logger,
                                                  logging_interval=logging_interval,
                                                  title=f'f1_score_by_'
                                                        f'{logging_interval}')


class F1ScoreNoDropoutMonitor(MetricMonitor):
    def __init__(self, stage='valid', logger =None, logging_interval=None,
                 title = None ):
        super(F1ScoreNoDropoutMonitor, self).__init__(stage=stage,
                                                  metric="f1_score_no_dropout",
                                                  logger=logger,
                                                  logging_interval=logging_interval,
                                                  title=f'f1_score_by_'
                                                        f'{logging_interval}',
                                              series='no_dropout')