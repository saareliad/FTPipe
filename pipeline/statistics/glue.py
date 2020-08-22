from .interface import Stats
from .utils import AverageMeter


def glue_compute_metrics_name(task_name):
    if task_name == "cola":
        return "mcc"
    elif task_name == "sst-2":
        return "acc"
    elif task_name == "mrpc":
        return "acc_and_f1"
    elif task_name == "sts-b":
        return "corr"
    elif task_name == "qqp":
        return "acc_and_f1"
    elif task_name == "mnli":
        return "mnli/acc"
    elif task_name == "mnli-mm":
        return "mnli-mm/acc"
    elif task_name == "qnli":
        return "acc"
    elif task_name == "rte":
        return "acc"
    elif task_name == "wnli":
        return "acc"
    elif task_name == "hans":
        return "acc"
    else:
        raise KeyError(task_name)


class GlueStats(Stats):
    """ Class to handle statistics collection for Glue Tasks """
    def __init__(self, record_loss_per_batch=False, is_last_partition=True):
        # Stats
        super().__init__(is_last_partition=is_last_partition)

        self.add_statistic(
            name="loss",
            meter=AverageMeter(),
            per_batch=record_loss_per_batch,
            per_epoch=not record_loss_per_batch,  # FIXME
            train=True,
            test=True)

        self.record_loss_per_batch = record_loss_per_batch

        # Glue results
        self.predictions = []
        self.label_ids = []

    def set_glue_task(self, task_name):
        self.task = task_name
        self.metric_name = glue_compute_metrics_name(task_name)

    def get_metric_for_early_stop(self):
        num_epochs = self.fit_res.num_epochs
        v = self.fit_res.glue_results[num_epochs][self.metric_name]
        return v

    def last_partition_on_epoch_end(self):
        super().last_partition_on_epoch_end()
        # NOTE: we can easly calc statistics like accuracy for train set but we don't do it.
        if not self.training:
            self.evaluate_glue()  # FIXME: set by dataset
            self.predictions.clear()  # Clear results for next time
            self.label_ids.clear()

    def get_epoch_info_str(self, is_train):
        # FIXME: in per-batch-loss it returns value for the last batch instead of for epoch!
        if is_train:
            name = "train"
            loss = self.fit_res.train_loss[-1]
        else:
            name = "valid"
            loss = self.fit_res.test_loss[-1]
            # TODO: glue eval

        return ' | {} loss {:5.2f}'.format(
            name, loss, name)


class NormGluestats(GlueStats):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

        self.add_statistic(
            name="grad_norm",
            meter=AverageMeter(),
            per_batch=self.record_loss_per_batch,
            per_epoch=not self.record_loss_per_batch,  # FIXME
            train=True,
            test=False)
        self.register_pipeline_per_stage_statistic("grad_norm")


class GlueDistanceNorm(NormGluestats):
    # FIXME: This whole chain of classes has HORRIBLE design. just implement it simple.

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.add_statistic(name="gap",
                           meter=AverageMeter(),
                           per_batch=self.record_loss_per_batch,
                           per_epoch=not self.record_loss_per_batch,
                           train=True,
                           test=False)
        self.register_pipeline_per_stage_statistic("gap")


class GlueDistance(GlueStats):
    # FIXME: This whole chain of classes has HORRIBLE design. just implement it simply.

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

        self.add_statistic(
            name="gap",
            meter=AverageMeter(),
            per_batch=self.record_loss_per_batch,
            per_epoch=not self.record_loss_per_batch,  # FIXME
            train=True,
            test=False)
        self.register_pipeline_per_stage_statistic("gap")
