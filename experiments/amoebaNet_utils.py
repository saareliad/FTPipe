import torch.nn as nn


"""Figure out what layers should have reductions."""
def calc_reduction_layers(num_cells, num_reduction_layers):
  reduction_layers = []
  for pool_num in range(1, num_reduction_layers + 1):
    layer_num = (float(pool_num) / (num_reduction_layers + 1)) * num_cells
    layer_num = int(layer_num)
    reduction_layers.append(layer_num)
  return reduction_layers


class net(nn.Module):
  def __init__(self,filter_size,prev_output = None):
    super().__init__()
    self.p1 = [nn.ReLU()]
    self.p2 = []
    self.p3 = []
    self.p4 = []
    self.filter_size = filter_size
    self.prev_output = prev_output

  def forward(self, x):

      self.p1.append(nn.Conv2d(in_channels=x.shape[1],out_channels=self.filter_size,kernel_size=1))
      self.p1.append(nn.BatchNorm2d(self.filter_size))


      output1 = x
      output2 = x

      #calculate output1
      for layer in p1:
          output1 = layer(output1)

      if !(self.prev_output is None):

          if curr_filter_shape != prev_filter_shape:

              self.p2.append(nn.ReLU())
              output2 = self.p2[len(self.p2)-1](output2)

              layers = factorized_reduction(prev_output,self.filter_size, stride=2)
              if len(layers) == 1:
                  for layer in layers[0]:
                      self.p2.append(layer)
                      output2 = self.p2[len(self.p2)-1](output2)
              else:
                  output3 = output2
                  output4 = output2

                  for layer in layers[0]:
                      self.p3.append(layer)
                      output3 = self.p3[len(self.p3)-1](output3)
                  for layer in layers[1]:
                      self.p4.append(layer)
                      output4 = self.p4[len(self.p4)-1](output4)
                  output2 = torch.cat((output3, output4), dim=1)
                  self.BN = nn.BatchNorm2d(output2.shape[1])
                  output2 = self.BN(output2)

          elif self.filter_size != prev_output.shape[1]:
              self.p2.append(nn.ReLU())
              output2 = self.p2[len(self.p2)-1](output2)
              self.p2.append(nn.Conv2d(in_channels=output2.shape[1], out_channels=self.filter_size, kernel_size=1))
              output2 = self.p2[len(self.p2)-1](output2)
              self.p2.append(nn.BatchNorm2d(self.filter_size))
              output2 = self.p2[len(self.p2)-1](output2)

      output1.append(output2)

      return input1



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

    model = net()
#calculate output
    model.add_layer(nn.ReLU(),1)
    model.add_layer( nn.Conv2d(in_channels=input.shape[1],out_channels=self._filter_size,kernel_size=1),1)
    model.add_layer( nn.BatchNorm2d(self._filter_size),1)

#calculate output2
    output2 = input
    if !(prev_output is None):
        if curr_filter_shape != prev_filter_shape:
            model.add_layer(nn.ReLU(),2)

            layers = factorized_reduction(self._filter_size, stride=2)

            if len(layers) == 1:
                for layer in layers[0]:
                    model.add_layer(layer,2)
            else:
                for layer in layers[0]:
                    model.add_layer(layer,3)
                for layer in layers[1]:
                    model.add_layer(layer,4)
        elif self._filter_size != prev_output.shape[1]:
            model.add_layer(nn.ReLU(),2)
            model.add_layer(nn.Conv2d(in_channels=output2.shape[1], out_channels=self._filter_size, kernel_size=1),2)
            model.add_layer(nn.BatchNorm2d(self._filter_size),2)


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
"""ssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss"""
      # Apply conv operations
     h1 = self._apply_operation(h1, operation_left,stride, original_input_left)
     h2 = self._apply_operation(h2,operation_right,stride,original_input_right)

    # Combine hidden states using 'add'.
     h = h1 + h2
    # Add hiddenstate to the list of hiddenstates we can choose from
     net.append(h)

     net = self._combine_unused_states(net)
    return net
"""ssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss"""


def factorized_reduction(output2, output_filters, stride):
  """Reduces the shape of net without information loss due to striding."""
  assert output_filters % 2 == 0, ('Need even number of filters when using this factorized reduction.')
  layers = [[]]
  if stride == 1:
    layers[0].append(nn.Conv2d(output2.shape[1], output_filters, 1))
    layers[0].append(nn.BatchNorm2d(output_filters))
    return layers

  layers.append([])
  # Skip path 1
  layers[0].append(nn.AvgPool2d(kernel_size=1,stride=stride))
  layers[0].append(nn.Conv2d(path1.shape[1],int(output_filters / 2), 1))
  # Skip path 2
  # First pad with 0's on the right and bottom, then shift the filter to
  # include those 0's that were added.
  layers[1].append(nn.ZeroPad2d((0, 1, 0, 1))) #take indices [:,:,1:,1:]
  layers[1].append(nn.AvgPool2d(kernel_size=1, stride=stride))
  layers[1].append(nn.Conv2d( path2.shape[1] , int(output_filters / 2), 1))
  # Concat and apply BN
  return layers
