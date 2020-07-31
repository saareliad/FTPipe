import shlex
import sys
from pytorch_Gpipe.model_partitioning.acyclic_partitioning import Objective, META_ALGORITH
import os
from shutil import copyfile
import argparse
import torch


class Parser(argparse.ArgumentParser):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
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

        analysis_args = self.add_argument_group("analysis_args")
        self._add_analysis_args(analysis_args)

        extra_args = self.add_argument_group("extra_args")
        self._extra(extra_args)

        self.set_defaults(**self._default_values())

    def _extra(self, group):
        pass

    def _default_values(self):
        return dict()

    def _add_model_args(self,group):
        raise NotImplementedError() 

    def _add_data_args(self,group):
        pass

    def _add_analysis_args(self, group):
        analysis_mode = group.add_mutually_exclusive_group()
        analysis_mode.add_argument('--no_analysis',
                            action='store_true',
                            default=False,
                            help="disable partition analysis")
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
        
        ratio_options = group.add_mutually_exclusive_group()
        ratio_options.add_argument("--bwd_to_fwd_ratio",
                            type=float,
                            default=-1,
                            help="bwd to fwd ratio for heuristics")
        ratio_options.add_argument(
            "--auto_infer_node_bwd_to_fwd_ratio", 
            action='store_true',
            default=False,
            help=
            "Automatically infer bwd to fwd ratio for nodes (computation)"
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
        group.add_argument(
            '--model_too_big',
            action='store_true',
            default=False,
            help=
            "if the model is too big run the whole partitioning process on CPU, "
            "and drink a cup of coffee in the meantime")
        group.add_argument('-p', '--n_partitions', type=int, default=4)
        group.add_argument('-o', '--output_file', default='')
        group.add_argument(
            '--n_iter',
            type=int,
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
        group.add_argument('--basic_blocks', nargs='*')
        group.add_argument(
            "--use_network_profiler",
            default=False,
            action="store_true",
            help=
            "wether to use the old network_profiler instead of the newer graph based profiler"
        )
        group.add_argument(
            "--disable_op_profiling",
            default=False,
            action="store_true",
            help="weheter to not profile ops when using the GraphProfiler")
        group.add_argument(
            "--use_METIS",
            default=False,
            action="store_true",
            help=
            "wether to use METIS partitioning instead of the acyclic partitioner"
        )
        group.add_argument(
            "--generate_model_parallel",
            action="store_true",
            default=False,
            help=
            "wether to generate a modelParallel version of the partitioning")
        group.add_argument(
            "--generate_explicit_del",
            action="store_true",
            default=False,
            help="wether to generate del statements in partitioned code")

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

        group.add_argument("-c", "--profiles_cache_name", default="", type=str, help="Profile cache to use in case of multiple runs")

        group.add_argument("--overwrite_profiles_cache", action="store_true", default=False, help="overwrite prilfes cache")

    def _add_METIS_args(self,group):
        group.add_argument("--metis_seed",
                                required=False,
                                type=int,
                                help="Random seed for Metis algorithm")
        group.add_argument(
            '--metis_compress',
            default=False,
            action='store_true',
            help="Compress")  # NOTE: this is differnt from default!
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

    def _add_acyclic_args(self,group):
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
            help="wether to use multilevel partitioning algorithm")
        group.add_argument("--objective",
                          choices=["edge_cut", "stage_time"],
                          default="edge_cut",
                          help="partitioning optimization objective")

    def parse_args(self,args=None, namespace=None):
        args = super().parse_args(args, namespace)

        args.acyclic_opt = self._acyclic_opts_dict_from_parsed_args(args)
        args.METIS_opt = self._metis_opts_dict_from_parsed_args(args)

        device = torch.device("cuda" if torch.cuda.is_available()
                          and not args.model_too_big else "cpu")
        args.device = device


        return args

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

        return {
            "epsilon": args.epsilon,
            "rounds": args.rounds,
            "allocated_seconds": args.allocated_seconds,
            "meta_algorithm": meta_algorithm,
            "objective": objective
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


def choose_blocks(model, args):
    blocks = dict()

    for m in model.modules():
        block = type(m)
        blocks[block.__name__] = block

    if args.basic_blocks is None:
        args.basic_blocks = []

    return tuple([blocks[name] for name in args.basic_blocks])


def prepend_line(filename, line):
    with open(filename, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(line.rstrip('\r\n') + '\n' + content)


def record_cmdline(output_file):
    """Add cmdline to generated python output file."""
    cmdline = " ".join(map(shlex.quote, sys.argv[:]))
    python_output_file = output_file + ".py"
    cmdline = '"""' + "AutoGenerated with:\n" + "python " + cmdline + "\n" + '"""'
    prepend_line(python_output_file, cmdline)


def run_x_tries_until_no_fail(func, number_of_tries, *args, **kw):
    count = 0
    success = False
    res = None

    while number_of_tries < 0 or count < number_of_tries:
        try:
            res = func(*args, **kw)
        except Exception as e:
            count += 1
            if count == number_of_tries - 1:
                print(
                    f"-E- run_x_tries_until_no_fail Failed after {count} raising last exception"
                )
                raise e
            continue

        success = True
        count += 1
        break

    print(f"running function got succcess={success} after {count} attempts")
    return res


def bruteforce_main(main, override_dicts=None, NUM_RUNS=2, TMP="/tmp/partitioning_outputs/") :
    # TODO: put all hyper parameters here, a dict for each setting we want to try.
    # d1 = dict(basic_blocks=[])
    # ovverride_dicts.append(d1)
    results = {}
    best = None
    
    if override_dicts is None:
        override_dicts = []

    if not override_dicts:
        override_dicts = [{}]

    DICT_PREFIX = "_d%d"
    current_dict_prefix = ""
    for i, override_dict in enumerate(override_dicts):
        if i > 0:
            current_dict_prefix = DICT_PREFIX.format(i)

        counter = 0
        while counter < NUM_RUNS:
            out = main(override_dict)
     
            try:
                os.makedirs(TMP, exist_ok=True)
                (analysis_result, args) = out

                name = args.output_file
                orig_name = name
                flag = False

                if name in results:
                    if name.endswith(".py"):
                        name = name[:-3]
                    flag = True

                while (name+".py" in results) or flag:
                    flag = False
                    name += f"_{counter}"
                
                name += current_dict_prefix
                new_path = os.path.join(TMP, name+".py")
                copyfile(orig_name+".py",  new_path)  # Save the last generated file

                results[name] = analysis_result

                if best is None:
                    best = (new_path, analysis_result)
                else:
                    if analysis_result > best[1]:
                        best = (new_path, analysis_result)

            except Exception as e:
                print("-E- running multiple times failed")
                raise e

            counter += 1

    print(results)
    print(f"best: {best}")
    copyfile(os.path.join(TMP, best[0]), orig_name+".py")
    print(f"-I- copied best to {orig_name}.py")

