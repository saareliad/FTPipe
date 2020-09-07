from .cv import CVStats, NormCVstats, CVDistanceNorm, CVDistance
from .glue import GlueStats, NormGluestats, GlueDistanceNorm, GlueDistance
from .interface import Stats
from .lm import LMStats, NormLMstats, LMDistanceNorm, LMDistance
from .squad import SquadStats, NormSquadstats, SquadDistanceNorm, SquadDistance

# TODO: remove the "record_loss_per_batch", it is not used mostly.
# TODO: option to record every X batches, when epoch is giant.

# TODO: change the way of getting statistics

AVAILBALE_STATS = {}


def register_statistics(name: str, stats_cls: Stats):
    AVAILBALE_STATS[name] = stats_cls
    AVAILBALE_STATS[name + "_loss_per_batch"] = stats_cls


#  is_last_partition=True
def get_statistics(name: str, *args, **kw) -> Stats:
    record_loss_per_batch = "loss_per_batch" in name
    st_cls = AVAILBALE_STATS.get(name)
    return st_cls(*args, record_loss_per_batch=record_loss_per_batch, **kw)


register_statistics("cv", CVStats)
register_statistics("cv_grad_norm", NormCVstats)
register_statistics("cv_theta_dist", CVDistance)
register_statistics("cv_dist_grad_norm", CVDistanceNorm)

register_statistics("lm", LMStats)
register_statistics("lm_grad_norm", NormLMstats)
register_statistics("lm_theta_dist", LMDistance)
register_statistics("lm_dist_grad_norm", LMDistanceNorm)

register_statistics("squad", SquadStats)
register_statistics("squad_grad_norm", NormSquadstats)
register_statistics("squad_theta_dist", SquadDistance)
register_statistics("squad_dist_grad_norm", SquadDistanceNorm)

register_statistics("glue", GlueStats)
register_statistics("glue_grad_norm", NormGluestats)
register_statistics("glue_theta_dist", GlueDistance)
register_statistics("glue_dist_grad_norm", GlueDistanceNorm)
