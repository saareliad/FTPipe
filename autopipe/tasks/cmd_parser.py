import argparse
from abc import ABC, abstractmethod
from argparse import Namespace
from gettext import gettext

import torch

from autopipe.autopipe.model_partitioning.acyclic_partitioning import Objective, META_ALGORITH, Constraint
from autopipe.autopipe.model_partitioning.bin_packing.partition_2dbinpack import ReminderPolicy, \
    SecondAndOnClusterPolicy


class Parser(argparse.ArgumentParser, ABC):
    """ArgumentParser for partitioning tasks,
        excluding tasks specific args (i.e model and data)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        model_args = self.add_argument_group("model_args")
        self._add_model_args(model_args)

        data_args = self.add_argument_group("data_args")
        self._add_data_args(data_args)

        partitioning_args = self.add_argument_group("partitioning_args")
        self._add_partitioning_args(partitioning_args)

        heuristics_args = self.add_argument_group("heuristics_args")
        self._add_heurisitcs_args(heuristics_args)

        METIS_args = self.add_argument_group("METIS_args")
        self._add_METIS_args(METIS_args)

        acyclic_args = self.add_argument_group("acyclic_args")
        self._add_acyclic_args(acyclic_args)

        binpack_args = self.add_argument_group("binpack_args")
        self._add_binpack_args(binpack_args)

        analysis_args = self.add_argument_group("analysis_args")
        self._add_analysis_args(analysis_args)

        extra_args = self.add_argument_group("extra_args")
        extra_args.add_argument("--debug", action="store_true", default=False)
        self._extra(extra_args)

        self.set_defaults(**self._default_values())

    @abstractmethod
    def _add_model_args(self, group):
        """add cmd args required for building the model that will be partitioned"""

    @abstractmethod
    def _add_data_args(self, group):
        """add cmd args required for providing input samples for partitioning"""

    def _add_analysis_args(self, group):
        analysis_mode = group.add_mutually_exclusive_group()
        analysis_mode.add_argument('--no_analysis',
                                   action='store_true',
                                   default=False,
                                   help="disable partition analysis")
        analysis_mode.add_argument('--analysis_only',
                                   action='store_true',
                                   default=False,
                                   help="run only analysis for partitioned model")
        group.add_argument(
            "--analysis_batch_size",
            default=32,
            type=int,
            help="batch size to use during the post partition analysis")

    def _add_heurisitcs_args(self, group):
        group.add_argument(
            '--bw',
            type=float,
            default=12,
            help=
            "data transfer rate between gpus in GBps (Gigabytes per second)")

        group.add_argument("--bwd_to_fwd_ratio",
                           type=float,
                           default=1,
                           help="bwd to fwd ratio for heuristics")
        group.add_argument(
            "--auto_infer_node_bwd_to_fwd_ratio",
            action='store_true',
            default=False,
            help=
            "Automatically infer bwd to fwd ratio for nodes (computation). Expected Ratio for edges should be given `by bwd_to_fwd_ratio`"
        )

        group.add_argument(
            "--penalize_non_tensors",
            action='store_true',
            default=False,
            help=
            "penalize edges with non tensor outputs by default no penalties are applied"
        )
        group.add_argument(
            "--weight_mult_factor",
            type=float,
            default=1e4,
            help=
            "a constant to multiply weights with (usefull if weights are really small)"
        )

        group.add_argument(
            "--edge_penalty",
            type=float,
            default=1e4,
            help=
            "multipicative penalty for edges if `penalize_non_tensors` is set"
        )

    def _add_partitioning_args(self, group):
        group.add_argument('-b',
                           '--partitioning_batch_size',
                           type=int,
                           default=128)
        group.add_argument('-p', '--n_partitions', type=int, default=4)
        group.add_argument('-o', '--output_file', default='')
        group.add_argument(
            '--n_iter',
            type=int,
            default=10,
            help=
            "number of iteration used in order to profile the network and run analysis"
        )
        group.add_argument(
            '--no_recomputation',
            action='store_true',
            default=False,
            help="whether to (not) use recomputation for the backward pass")
        group.add_argument(
            "--depth",
            default=10000,
            type=int,
            help="the depth in which we will partition the model")
        group.add_argument('--basic_blocks', nargs='*', default=[])
        group.add_argument(
            "--use_network_profiler",
            default=False,
            action="store_true",
            help=
            "whether to use the old network_profiler instead of the newer graph based profiler"
        )
        group.add_argument(
            "--disable_op_profiling",
            default=False,
            action="store_true",
            help="weheter to not profile ops when using the GraphProfiler")
        group.add_argument("--partitioning_method", "-m", choices=["ACYCLIC", "METIS", "2DBIN"], default="ACYCLIC")
        group.add_argument(
            "--generate_model_parallel",
            action="store_true",
            default=False,
            help=
            "whether to generate a modelParallel version of the partitioning")
        group.add_argument(
            "--generate_explicit_del",
            action="store_true",
            default=False,
            help="whether to generate del statements in partitioned code")
        group.add_argument(
            "--no_activation_propagation",
            action="store_true",
            default=False,
            help="whether to not propgate activations in the pipeline, and having direct sends instead"
        )

        group.add_argument("-a",
                           "--async_pipeline",
                           default=False,
                           action="store_true",
                           help="Do analysis for async pipeline")
        group.add_argument("--dot",
                           default=False,
                           action="store_true",
                           help="Save and plot it using graphviz")

        group.add_argument(
            "--save_memory_mode",
            default=False,
            action="store_true",
            help="Save memory during profiling by storing everything on cpu," +
                 " but sending each layer to GPU before the profiling.")

        group.add_argument("--force_no_recomputation_scopes", nargs="*", default=[])

        group.add_argument("-c", "--profiles_cache_name", default="", type=str,
                           help="Profile cache to use in case of multiple runs")

        group.add_argument("--overwrite_profiles_cache", action="store_true", default=False,
                           help="overwrite profile cache")

    # TODO should use sub parsers as those 2 groups are mutaully exclusive
    def _add_METIS_args(self, group):
        group.add_argument("--metis_attempts", type=int, default=10,
                           help="number of attempts for running the METIS partitioning algorithm")
        group.add_argument("--metis_verbose_on_error", action="store_true", default=False,
                           help="whether to print the cause of the error")
        group.add_argument("--metis_seed",
                           required=False,
                           type=int,
                           help="Random seed for Metis algorithm")
        group.add_argument(
            '--metis_compress',
            default=False,
            action='store_true',
            help="Compress")  # NOTE: this is different from default!
        group.add_argument(
            '--metis_niter',
            type=int,
            help=
            "Specifies the number of iterations for the refinement algorithms at each stage of the uncoarsening process."
            "Default is 10.")
        group.add_argument(
            '--metis_nseps',
            type=int,
            help=
            "Specifies the number of different separators that it will compute at each level of nested dissection."
            "The final separator that is used is the smallest one. Default is 1."
        )
        group.add_argument(
            "--metis_ncuts",
            type=int,
            help=
            "Specifies the number of different partitionings that it will compute."
            " The final partitioning is the one that achieves the best edgecut or communication volume."
            "Default is 1.")
        group.add_argument(
            '--metis_dbglvl',
            type=int,
            help="Metis debug level. Refer to the docs for explanation")
        group.add_argument(
            '--metis_objtype',
            type=int,
            help=
            "Extra objective type to miminize (0: edgecut, 1: vol, default: edgecut)"
        )
        group.add_argument(
            '--metis_contig',
            default=False,
            action='store_true',
            help="A boolean to create contigous partitions."
            # see http://glaros.dtc.umn.edu/gkhome/metis/metis/faq"
        )

    def _add_binpack_args(self, group):
        group.add_argument("--n_clusters",
                           default=2,
                           type=int,
                           help="number of clusters in the model")
        group.add_argument("--analyze_n_clusters", action="store_true", default=False,
                           help="analyze number of clusters")

        group.add_argument("--reminder_policy", type=str,
                           choices=list(ReminderPolicy._value2member_map_.keys()),
                           default="last",
                           help=f"Policy for reminder items in cluster, {ReminderPolicy._value2member_map_}")

        group.add_argument("--second_and_on_cluster_policy", type=str,
                           choices=list(SecondAndOnClusterPolicy._value2member_map_.keys()),
                           default="best_fit",
                           help=f"Policy for 2nd and on cluster {SecondAndOnClusterPolicy._value2member_map_}")

        group.add_argument("--THRESHOLD", type=float, default=0,
                           help="values <= threshold will be contagious with closest stage")

    def _add_acyclic_args(self, group):
        group.add_argument("--epsilon",
                           default=0.1,
                           type=float,
                           help="imbalance factor")
        group.add_argument("--rounds",
                           default=10,
                           type=int,
                           help="number of optimization rounds default is 10")
        group.add_argument(
            "--allocated_seconds",
            default=20,
            type=int,
            help=
            "run time allocated to the partitioning algorithm default is 20 seconds"
        )
        group.add_argument(
            "--multilevel",
            action="store_true",
            default=False,
            help="whether to use multilevel partitioning algorithm")
        group.add_argument("--objective",
                           choices=["edge_cut", "stage_time"],
                           default="stage_time",
                           help="partitioning optimization objective")
        group.add_argument("--constraint",
                           choices=["time", "memory"],
                           default="time",
                           help="partitioning constraint")
        group.add_argument("--maximum_constraint_value",
                           required=False,
                           type=float,
                           default=None,
                           help=
                           "maximum constraint value a single stage can have,for example for memory constraint this is the maximum number of parameters a stage can have")

    def _extra(self, group):
        """add any extra cmd args which are task specific"""

    def _default_values(self):
        """ provide default cmd args values"""
        return dict()

    def _post_parse(self, args, argv):
        """handler for parsing unkonwn cmd args
        """
        # NOTE currently the use case is mainly for megatron in order to pass arguments to their parser

        # taken from argparse.ArgumentParser.parse_args
        if argv:
            msg = gettext('unrecognized arguments: %s')
            self.error(msg % ' '.join(argv))
        return args

    def _auto_file_name(self, args) -> str:
        """ provide a file name if args.output_file was not specified"""

        return ""

    def parse_args(self, args=None, namespace=None) -> Namespace:
        args, extra = super().parse_known_args(args, namespace)
        args.acyclic_opt = self._acyclic_opts_dict_from_parsed_args(args)
        args.METIS_opt = self._metis_opts_dict_from_parsed_args(args)
        args.binpack_opt = self._binpack_opts_dict_from_parsed_args(args)

        if not args.output_file:
            args.output_file = self._auto_file_name(args)
        if args.output_file.endswith(".py"):
            args.output_file = args.output_file[:-3]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.device = device

        args.force_no_recomputation_scopes_fn = lambda scope: any(
            s in scope for s in args.force_no_recomputation_scopes)

        return self._post_parse(args, extra)

    @staticmethod
    def _acyclic_opts_dict_from_parsed_args(args):
        """ build acyclic partitioner options """

        if args.objective == "edge_cut":
            objective = Objective.EDGE_CUT
        else:
            objective = Objective.STAGE_TIME

        if args.multilevel:
            meta_algorithm = META_ALGORITH.MULTI_LEVEL
        else:
            meta_algorithm = META_ALGORITH.SINGLE_LEVEL

        if args.constraint == "time":
            constraint = Constraint.TIME
        else:
            constraint = Constraint.MEMORY

        return {
            "epsilon": args.epsilon,
            "rounds": args.rounds,
            "allocated_seconds": args.allocated_seconds,
            "meta_algorithm": meta_algorithm,
            "objective": objective,
            "constraint": constraint,
            "maximum_constraint_value": args.maximum_constraint_value
        }

    @staticmethod
    def _metis_opts_dict_from_parsed_args(args):
        """ build metis options """

        #     {'ptype': -1,
        #  'objtype': -1,
        #  'ctype': -1,
        #  'iptype': -1,
        #  'rtype': -1,
        #  'ncuts': -1,
        #  'nseps': -1,
        #  'numbering': -1,
        #  'niter': -1, # default is 10
        #  'seed': -1,
        #  'minconn': True,
        #  'no2hop': True,
        #  'contig': True,
        #  'compress': True,
        #  'ccorder': True,
        #  'pfactor': -1,
        #  'ufactor': -1,
        #  '_dbglvl': -1,
        #  }

        # We can set to None to get the default
        # See : https://github.com/networkx/networkx-metis/blob/master/nxmetis/enums.py
        METIS_opt = {
            'verbose_on_error': getattr(args, 'metis_verbose_on_error', False),
            'attempts': getattr(args, "metis_attempts", 1000),
            'seed': getattr(args, "metis_seed", None),
            'nseps': getattr(args, "nseps", None),
            'niter': getattr(args, "metis_niter", None),
            'compress': getattr(args, "metis_compress",
                                None),  # NOTE: this is differnt from default!
            'ncuts': getattr(args, "metis_ncuts", None),
            # 0, edgecut, 1 Vol minimization! # NOTE: this is differnt from default edgecut.
            'objtype': getattr(args, 'metis_objtype', None),
            'contig': getattr(args, 'metis_contig', None),
            # NOTE: default is -1, # TODO: add getattr getattr(args, "metis_dbglvl", None),
            '_dbglvl': 1  # TODO: can't make it print...
        }

        return METIS_opt

    def _binpack_opts_dict_from_parsed_args(self, args):
        d = dict()
        d['n_clusters'] = args.n_clusters
        d['analyze_n_clusters'] = args.analyze_n_clusters

        if hasattr(args, "second_and_on_cluster_policy"):
            d['second_and_on_cluster_policy'] = args.second_and_on_cluster_policy

        if hasattr(args, "reminder_policy"):
            d['reminder_policy'] = args.reminder_policy

        d['THRESHOLD'] = args.THRESHOLD

        return d