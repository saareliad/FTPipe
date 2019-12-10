import torch
from models.normal.dummyDAG import DummyDAG
from models.partitioned.dDAG import createConfig
from pipeline.util import create_buffer_configs
from pipeline.GeneralCommHandler import CommunicationHandler, createCommParams

if __name__ == "__main__":
    # visual check to see if the config turns out as intended
    model = DummyDAG()
    config = createConfig(model, cpu=True)
    buffer_config = create_buffer_configs(torch.randn(100, 200), config)

    print("buffer info")
    for k, v in buffer_config.items():
        print(k, v)
    print()

    for pIdx in list(range(5))+[5, 6]:
        print(f"partiton {pIdx}")
        policy, i, o, t = createCommParams(pIdx, 'mpi', config,
                                           buffer_config, cpu=True)

        print("inputs")
        for e in i:
            print(f"{e[0],e[1],e[2].shape,e[2].dtype,e[3],e[4]}")

        print("outputs")
        for e in o:
            print(f"{e[0],e[1],e[2].shape,e[2].dtype,e[3],e[4]}")

        print()

    print(f"total_tags {t}")
