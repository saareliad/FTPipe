from .interface import Stats
from .utils import AverageMeter


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

    def get_metric_for_early_stop(self):
        num_epochs = self.fit_res.num_epochs
        # FIXME: by task name
        v = self.fit_res.glue_results[num_epochs]['acc']
        return v

    def last_partition_on_epoch_end(self):
        super().last_partition_on_epoch_end()
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
