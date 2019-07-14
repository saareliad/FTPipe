from collections import namedtuple
from functools import reduce
import operator as op
import ctypes
from enum import IntEnum
from ctypes import POINTER as ptr, byref
import os
import typing
from typing import List, Tuple, Dict, Union, Callable
import platform

# metis lib loading
if 'Win' in platform.system():
    metis_path = os.path.dirname(os.path.realpath(__file__))+"/libmetis.dll"
elif 'Lin' in platform.system():
    metis_path = os.path.dirname(os.path.realpath(__file__))+"/libmetis.so"

if not os.path.exists(metis_path):
    raise FileNotFoundError(
        "the metis library could not be found please follow the build notes for further instructions")

metisLib = ctypes.CDLL(metis_path)

# -------------------------------------------------------------------------
# type declarations and constants
# -------------------------------------------------------------------------
idx_t = ctypes.c_int64
real_t = ctypes.c_double
METIS_NOPTIONS = 40
METIS_Graph = namedtuple('METIS_Graph',
                         'nvtxs ncon xadj adjncy vwgt vsize adjwgt')


# -------------------------------------------------------------------------
# Enum type definitions
# -------------------------------------------------------------------------
class _MetisEnum(IntEnum):
    """A ctypes-compatible IntEnum superclass."""
    @classmethod
    def from_param(cls, obj):
        return int(obj)


# Return codes
class rstatus_et(_MetisEnum):
    METIS_OK = 1            # Returned normally
    METIS_ERROR_INPUT = -2  # Returned due to erroneous inputs and/or options
    METIS_ERROR_MEMORY = -3  # Returned due to insufficient memory
    METIS_ERROR = -4  # Some other errors


# Operation type codes
class moptype_et (_MetisEnum):

    METIS_OP_PMETIS = 0
    METIS_OP_KMETIS = 1
    METIS_OP_OMETIS = 2


# Options codes(i.e., options[])
class moptions_et(_MetisEnum):
    METIS_OPTION_PTYPE = 0
    METIS_OPTION_OBJTYPE = 1
    METIS_OPTION_CTYPE = 2
    METIS_OPTION_IPTYPE = 3
    METIS_OPTION_RTYPE = 4
    METIS_OPTION_DBGLVL = 5
    METIS_OPTION_NITER = 6
    METIS_OPTION_NCUTS = 7
    METIS_OPTION_SEED = 8
    METIS_OPTION_MINCONN = 9
    METIS_OPTION_CONTIG = 10
    METIS_OPTION_COMPRESS = 11
    METIS_OPTION_CCORDER = 12
    METIS_OPTION_PFACTOR = 13
    METIS_OPTION_NSEPS = 14
    METIS_OPTION_UFACTOR = 15
    METIS_OPTION_NUMBERING = 16

    # Used for command-line parameter purposes
    METIS_OPTION_HELP = 18
    METIS_OPTION_TPWGTS = 19
    METIS_OPTION_NCOMMON = 20
    METIS_OPTION_NOOUTPUT = 21
    METIS_OPTION_BALANCE = 22
    METIS_OPTION_GTYPE = 23
    METIS_OPTION_UBVEC = 24


# Partitioning Schemes
class mptype_et (_MetisEnum):
    METIS_PTYPE_RB = 1
    METIS_PTYPE_KWAY = 1


# Graph types for meshes
class mgtype_et(_MetisEnum):
    METIS_GTYPE_DUAL = 0
    METIS_GTYPE_NODAL = 1


# Coarsening Schemes
class mctype_et(_MetisEnum):
    METIS_CTYPE_RM = 0
    METIS_CTYPE_SHEM = 1


# Initial partitioning schemes
class miptype_et(_MetisEnum):
    METIS_IPTYPE_GROW = 0
    METIS_IPTYPE_RANDOM = 1
    METIS_IPTYPE_EDGE = 2
    METIS_IPTYPE_NODE = 3
    METIS_IPTYPE_METISRB = 4


# Refinement schemes
class mrtype_et(_MetisEnum):
    METIS_RTYPE_FM = 0
    METIS_RTYPE_GREEDY = 1
    METIS_RTYPE_SEP2SIDED = 2
    METIS_RTYPE_SEP1SIDED = 3


# Debug Levels
class mdbglvl_et(_MetisEnum):
    METIS_DBG_INFO = 1         # Shows various diagnostic messages
    METIS_DBG_TIME = 2         # Perform timing analysis
    METIS_DBG_COARSEN = 4      # Show the coarsening progress
    METIS_DBG_REFINE = 8       # Show the refinement progress
    METIS_DBG_IPART = 16       # Show info on initial partitioning
    METIS_DBG_MOVEINFO = 32    # Show info on vertex moves during refinement
    METIS_DBG_SEPINFO = 64     # Show info on vertex moves during sep refinement
    METIS_DBG_CONNINFO = 128   # Show info on minimization of subdomain connectivity
    METIS_DBG_CONTIGINFO = 256  # Show info on elimination of connected components
    METIS_DBG_MEMORY = 2048    # Show info related to wspace allocation
    METIS_DBG_ALL = sum(2**i for i in list(range(9))+[11])


# Types of objectives
class mobjtype_et(_MetisEnum):
    METIS_OBJTYPE_CUT = 0
    METIS_OBJTYPE_VOL = 1
    METIS_OBJTYPE_NODE = 2


# -------------------------------------------------------------------------
# errors and error handler
# -------------------------------------------------------------------------
def _error_handler(METIS_result):
    if METIS_result != rstatus_et.METIS_OK:
        if METIS_result == rstatus_et.METIS_ERROR_INPUT:
            raise METIS_InputError
        if METIS_result == rstatus_et.METIS_ERROR_MEMORY:
            raise METIS_MemoryError
        if METIS_result == rstatus_et.METIS_ERROR:
            raise METIS_OtherError
        raise RuntimeError("weird status error")
    return METIS_result


class METIS_Error(Exception):
    pass


class METIS_MemoryError(METIS_Error, MemoryError):
    pass


class METIS_InputError(METIS_Error, ValueError):
    pass


class METIS_OtherError(METIS_Error):
    pass


# -------------------------------------------------------------------------
# lib function declarations and argtypes
# -------------------------------------------------------------------------
_part_graph = metisLib.METIS_PartGraphKway
_part_graph_recursive = metisLib.METIS_PartGraphRecursive
_SetDefaultOptions = metisLib.METIS_SetDefaultOptions

METIS_PartGraphKway_args = [ptr(idx_t), ptr(idx_t), ptr(idx_t), ptr(idx_t),
                            ptr(idx_t), ptr(idx_t), ptr(
    idx_t), ptr(idx_t), ptr(real_t),
    ptr(real_t), ptr(idx_t), ptr(idx_t), ptr(idx_t)
]
METIS_PartGraphRecursive_args = [ptr(idx_t), ptr(idx_t), ptr(idx_t), ptr(idx_t),
                                 ptr(idx_t), ptr(idx_t), ptr(
    idx_t), ptr(idx_t),
    ptr(real_t), ptr(real_t), ptr(idx_t),
    ptr(idx_t), ptr(idx_t)
]
METIS_SetDefaultOptions_args = [ptr(idx_t)]

_part_graph.argtypes = METIS_PartGraphKway_args
_part_graph_recursive.argtypes = METIS_PartGraphRecursive_args
_SetDefaultOptions.argtypes = METIS_SetDefaultOptions_args


# -------------------------------------------------------------------------
# help functions
# -------------------------------------------------------------------------
def _adjlist_to_metis(adjlist, nodew=None, nodesz=None):
    """
    :param adjlist: A list of tuples. Each list element represents a node or vertex
      in the graph. Each item in the tuples represents an edge. These items may be
      single integers representing neighbor index, or they may be an (index, weight)
      tuple if you want weighted edges. Default weight is 1 for missing weights.
      The graph must be undirected, and each edge must be represented twice (once for
      each node). The weights should be identical, if provided.

    :param nodew: is a list of node weights, and must be the same size as `adjlist` if given.
      If desired, the elements of `nodew` may be tuples of the same size (>= 1) to provided
      multiple weights for each node.

    :param nodesz: is a list of node sizes. These are relevant when doing volume-based
      partitioning.

    Note that all weights and sizes must be non-negative integers.
    """
    n = len(adjlist)
    m2 = sum(map(len, adjlist))

    xadj = (idx_t*(n+1))()
    adjncy = (idx_t*m2)()
    adjwgt = (idx_t*m2)()
    seen_adjwgt = False

    ncon = idx_t(1)
    if nodew:
        if isinstance(nodew[0], int):
            vwgt = (idx_t*n)(*nodew)
        else:
            nw = len(nodew[0])
            ncon = idx_t(nw)
            vwgt = (idx_t*(nw*n))(*reduce(op.add, nodew))
    else:
        vwgt = None

    if nodesz:
        vsize = (idx_t*n)(*nodesz)
    else:
        vsize = None

    xadj[0] = 0
    edge_idx = 0
    for i, adj in enumerate(adjlist):
        for j in adj:
            try:
                adjncy[edge_idx], adjwgt[edge_idx] = j
                seen_adjwgt = True
            except TypeError:
                adjncy[edge_idx], adjwgt[edge_idx] = j, 1
            edge_idx += 1
        xadj[i+1] = edge_idx

    # pass edge wieghts only they are being used
    if not seen_adjwgt:
        adjwgt = None

    return METIS_Graph(idx_t(n), ncon, xadj, adjncy, vwgt, vsize, adjwgt)


def _set_options(**options):
    '''
    create a metis options object
    given options will be used, and all other will be set to their default value

    options:
        a dictionary containing pairs of option value pairs
        for example {'dbglvl':mdbglvl_et.METIS_DBG_ALL}
    '''
    opts_arr = (idx_t*METIS_NOPTIONS)()
    _SetDefaultOptions(opts_arr)
    if options != None:
        for key, value in options.items():
            opt_key = f"METIS_OPTION_{key.upper()}"
            try:
                moptions_et[opt_key]
            except KeyError as _:
                print("invalid metis option")
            opts_arr[moptions_et[opt_key]] = value
    return opts_arr


# -------------------------------------------------------------------------
# python API
# -------------------------------------------------------------------------
def METIS_partition(adjlist, nparts=2, tpwgts=None, ubvec=None, algorithm='metis', **opts) -> Tuple[int, List[int]]:
    """
    Perform graph partitioning using k-way or recursive methods.

    Returns a 2-tuple `(parts,objval)`, where `parts` is a list of
    partition indices corresponding and `objval` is the value of
    the objective function that was minimized (either the edge cuts
    or the total volume).

    :param adjlist: an adjacency list describing the graph

      See :func:`_adjlist_to_metis` for information on the use of adjacency lists.
      The extra ``nodew`` and ``nodesz`` keyword arguments of that function may be given
      directly to this function and will be forwarded to the converter.
      Alternatively, a dictionary can be provided as ``graph`` and its items
      will be passed as keyword arguments.

    :param nparts: The target number of partitions. You might get fewer.
    :param tpwgts: Target partition weights. For each partition, there should
      be one (float) weight for each node constraint. That is, if `nparts` is 3 and
      each node of the graph has two weights, then tpwgts might look like this::

        [(0.5, 0.1), (0.25, 0.8), (0.25, 0.1)]

      This list may be provided flattened. The internal tuples are for convenience.
      The partition weights for each constraint must sum to 1.
    :param ubvec: The load imalance tolerance for each node constraint. Should be
      a list of floating point values each greater than 1.

    :param algorithm: Determines the partitioning algorithm to use can be either
        metis for METIS_PartGraphKway and metis_recursive for METIS_PartGraphRecursive 

    Any additional METIS options may be specified as keyword parameters.
    See the METIS manual for specific meaning of each option.
    """
    if(nparts <= 1):
        raise ValueError("nparts must be greater than 1")

    nodesz = opts.pop('nodesz', None)
    nodew = opts.pop('nodew', None)
    graph = _adjlist_to_metis(adjlist, nodew=nodew, nodesz=nodesz)

    options = _set_options(**opts)
    if tpwgts and not isinstance(tpwgts, ctypes.Array):
        if isinstance(tpwgts[0], (tuple, list)):
            tpwgts = reduce(op.add, tpwgts)
        tpwgts = (real_t*len(tpwgts))(*tpwgts)
    if ubvec and not isinstance(ubvec, ctypes.Array):
        ubvec = (real_t*len(ubvec))(*ubvec)

    if tpwgts:
        assert len(tpwgts) == nparts * graph.ncon.value
    if ubvec:
        assert len(ubvec) == graph.ncon.value

    nparts_var = idx_t(nparts)

    objval = idx_t()
    partition = (idx_t*graph.nvtxs.value)()

    args = (byref(graph.nvtxs), byref(graph.ncon), graph.xadj,
            graph.adjncy, graph.vwgt, graph.vsize, graph.adjwgt,
            byref(nparts_var), tpwgts, ubvec, options,
            byref(objval), partition)

    if algorithm == "metis_recursive":
        res = _part_graph_recursive(*args)
    elif algorithm == "metis":
        res = _part_graph(*args)
    else:
        raise NotImplementedError("bad algorithm")
    _error_handler(res)

    return list(partition), objval.value
