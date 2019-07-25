import torch.nn as nn


"""Figure out what layers should have reductions."""
def calc_reduction_layers(num_cells, num_reduction_layers):
  reduction_layers = []
  for pool_num in range(1, num_reduction_layers + 1):
    layer_num = (float(pool_num) / (num_reduction_layers + 1)) * num_cells
    layer_num = int(layer_num)
    reduction_layers.append(layer_num)
  return reduction_layers


""" let net be a list of hiddenstates"""
class BaseCell(object):
  """Base Cell class that is used as a 'layer' in image architectures.
  Args:
    num_conv_filters: The number of filters for each convolution operation.
    operations: List of operations that are performed in the AmoebaNet Cell in
      order.
    used_hiddenstates: Binary array that signals if the hiddenstate was used
      within the cell. This is used to determine what outputs of the cell
      should be concatenated together.
    hiddenstate_indices: Determines what hiddenstates should be combined
      together with the specified operations to create the AmoebaNet cell.
  """

  def __init__(self, num_conv_filters, operations, used_hiddenstates,
               hiddenstate_indices, drop_path_keep_prob, total_num_cells,
               total_training_steps,in_channels):
    self._num_conv_filters = num_conv_filters
    self._operations = operations
    self._used_hiddenstates = used_hiddenstates
    self._hiddenstate_indices = hiddenstate_indices
    self._drop_path_keep_prob = drop_path_keep_prob
    self._total_num_cells = total_num_cells
    self._total_training_steps = total_training_steps


# input shape N,C,H,W
  def __call__(self, input ,prev_output,filter_scaling=1, stride=1,, cell_num=-1):
    """Runs the conv cell."""
    self._cell_num = cell_num
    self._filter_scaling = filter_scaling
    self._filter_size = int(self._num_conv_filters * filter_scaling)


#calculate output
    output = nn.ReLU()(input)
    output = nn.Conv2d(in_channels=input.shape[1],out_channels=self._filter_size,kernel_size=1)(output)
    output = nn.BatchNorm2d(self._filter_size)(output)
#calculate output2
    output2 = input
    if !(prev_output is None):
        if curr_filter_shape != prev_filter_shape:
            output2 = nn.ReLU()(prev_output)
            output2 = factorized_reduction(output2, self._filter_size, stride=2)
        elif self._filter_size != prev_output.shape[1]:
            output2 = nn.ReLU()(prev_output)
            output2 = nn.Conv2d(in_channels=output2.shape[1], out_channels=self._filter_size, kernel_size=1)(output2)
            output2 = batch_norm(self._filter_size)(output2)
     output.append(output2)

"""ssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss"""
    i = 0
    for iteration in range(5):
      left_hiddenstate_idx = self._hiddenstate_indices[i]
      right_hiddenstate_idx = self._hiddenstate_indices[i + 1]

      original_input_left = left_hiddenstate_idx < 2
      original_input_right = right_hiddenstate_idx < 2

      h1 = net[left_hiddenstate_idx]
      h2 = net[right_hiddenstate_idx]

      operation_left = self._operations[i]
      operation_right = self._operations[i+1]
      i += 2
      # Apply conv operations
      with tf.variable_scope('left'):
          h1 = self._apply_operation(h1, operation_left,
                                       stride, original_input_left)
      with tf.variable_scope('right'):
          h2 = self._apply_operation(h2, operation_right,
                                       stride, original_input_right)

          # Combine hidden states using 'add'.
      with tf.variable_scope('combine'):
          h = h1 + h2
          # Add hiddenstate to the list of hiddenstates we can choose from
          net.append(h)

      with tf.variable_scope('cell_output'):
        net = self._combine_unused_states(net)
    return net
"""ssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss"""


def factorized_reduction(output2, output_filters, stride):
  """Reduces the shape of net without information loss due to striding."""
  assert output_filters % 2 == 0, ('Need even number of filters when using this factorized reduction.')
  if stride == 1:
    output2 = nn.Conv2d(output2.shape[1], output_filters, 1)(output2)
    output2 = batch_norm(output_filters)(output2)
    return output2
  # Skip path 1
  path1 = nn.AvgPool2d(kernel_size=1,stride=stride)(output2)
  path1 = nn.Conv2d(path1.shape[1],int(output_filters / 2), 1)(path1)
  # Skip path 2
  # First pad with 0's on the right and bottom, then shift the filter to
  # include those 0's that were added.
  path2 = nn.ZeroPad2d((0, 1, 0, 1))(output2)[:, :, 1:, 1:]
  path2 = nn.AvgPool2d(kernel_size=1, stride=stride)(path2)
  path2 = nn.Conv2d( path2.shape[1] , int(output_filters / 2), 1)(path2)
  # Concat and apply BN
  final_path = torch.cat((path2, path1), dim=1)
  final_path = batch_norm(final_path.shape[1])(final_path)
  return final_path
