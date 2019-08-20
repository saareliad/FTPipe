from amoebaNet import AmoebaNet
import torch


class hparams(object):
    def __init__(self):
        "amobanet-a params"
        self.reduction_size = 256
        self.normal_cell_operations = ['avg_pool_3x3', 'max_pool_3x3', 'separable_3x3_2', 'none',
                                       'none', 'avg_pool_3x3', 'separable_3x3_2', 'separable_5x5_2',
                                       'avg_pool_3x3', 'separable_3x3_2']
        self.normal_cell_used_hiddenstates = [1, 0, 1, 0, 0, 1, 0]
        self.normal_cell_hiddenstate_indices = [0, 0, 1, 1, 0, 1, 0, 2, 5, 0]
        self.reduction_cell_operations = ['separable_3x3_2', 'avg_pool_3x3', 'max_pool_3x3',
                                          'separable_7x7_2', 'max_pool_3x3', 'max_pool_3x3',
                                          'separable_3x3_2', '1x7_7x1', 'avg_pool_3x3',
                                          'separable_7x7_2']
        self.reduction_cell_used_hiddenstates = [1, 1, 0, 0, 0, 0, 0]
        self.reduction_cell_hiddenstate_indices = [
            1, 0, 0, 2, 1, 0, 4, 0, 1, 0]
        self.drop_connect_keep_prob = 0.7
        self.num_cells = 3
        self.num_total_steps = 0
        self.num_reduction_layers = 2
        self.use_aux_head = True
        self.dense_dropout_keep_prob = 0.5
        self.stem_reduction_size = 32
        self.aux_scaling = 0.4


if __name__ == '__main__':
    x = torch.zeros((32, 3, 256, 256))
    hp = hparams()
    num_classes = 1001
    ann = AmoebaNet(inputs=x, num_classes=num_classes,
                    is_training=True, hparams=hp)
    # ann(x)
