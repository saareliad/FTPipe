from pprint import pprint

# from models.partitioned.t5_3b_tied_lmheads_320_8_8p_bw12_squad1_virtual_stages import *
# from models.partitioned.t5_3b_tied_lmheads_64_4_8p_bw12_squad1_acyclic import *
from models.partitioned.t5_3b_tied_lmheads_64_4_8p_bw12_squad1_pipedream import *

if __name__ == '__main__':


    for i, v in list(locals().items()):
        i: str
        if not i.startswith("Partition"):
            continue
        print(i)
        pprint(v.LAYER_SCOPES)
        pprint(v.TENSORS)
        print()

