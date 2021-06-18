import torch
from torch.utils.data import DataLoader

from autopipe.tasks import Parser, PartitioningTask, register_task
from models.normal.cep import Net, Dataset


# # N = number of nodes in graph, K = clique size
# # C = constant as defined in the paper, samples_num = arbitrary high number
# N, K, C, samples_num = 361, 18, 20000, 1e11
# # Loss, Optimizers etc..
# loss_func = nn.BCEWithLogitsLoss()
#
# model = Net(N, C)
# dataset = Dataset(N, K, samples_num)
# optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
#
# print(summary(model, input_size=(N * (N - 1) // 2,)))


class CEPParser(Parser):
    def _add_model_args(self, group):
        group.add_argument("--N", type=int, default=361)
        group.add_argument("--C", type=int, default=10000)

    def _add_data_args(self, group):
        group.add_argument("--K", type=int, default=18)
        group.add_argument("--samples_num", type=int, default=1e11)

    def _default_values(self):
        return {
            "n_iter": 1,
            "n_partitions": 4,
            "bw": 12,
            "partitioning_batch_size": 1,
            "analysis_batch_size": 1,

            # "basic_blocks": ["T5Attention"]
        }

    def _auto_file_name(self, args) -> str:
        bw_str = str(args.bw).replace(".", "_")
        model_str = str("cep_net").replace("-", "_")

        model_str += f"N{args.N}_C{args.C}"
        output_file = f"{args.output_file}{model_str}_{args.n_partitions}p_bw{bw_str}"

        if args.async_pipeline:
            output_file += "_async"

        m = args.partitioning_method.lower()
        tmp = m if m != "2dbin" else "virtual_stages"
        output_file += f"_{tmp}"

        return output_file


class CEPPartitioningTask(PartitioningTask):
    def __init__(self, args) -> None:
        super().__init__(args)

    @property
    def batch_dim(self) -> int:
        return 0

    def register_functions(self):
        pass

    #     register_new_explicit_untraced_function(operator.is_, operator)
    #     register_new_explicit_untraced_function(operator.is_not, operator)
    #     register_new_traced_function(math.log, math)
    #     register_new_traced_function(torch.einsum, torch)

    def get_model(self, args) -> torch.nn.Module:
        return Net(n=args.N, c=args.C)

    def get_input(self, args, analysis=False):
        dataset = Dataset(n=args.N, k=args.K, max_samples_num=args.samples_num)
        batch_size = args.analysis_batch_size if analysis else args.partitioning_batch_size
        loader = DataLoader(dataset, batch_size=batch_size)
        return next(iter(loader))[0]


register_task("cep", CEPParser, CEPPartitioningTask)
