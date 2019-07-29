import torch.nn as nn
import torch
from torch.distributions import uniform


def _operation_to_filter_shape(operation):
  splitted_operation = operation.split('x')
  filter_shape = int(splitted_operation[0][-1])
  assert filter_shape == int(splitted_operation[1][0]), 'Rectangular filters not supported.'
  return filter_shape


def _operation_to_num_layers(operation):
  splitted_operation = operation.split('_')
  if 'x' in splitted_operation[-1]:
    return 1
  return int(splitted_operation[-1])


def _operation_to_info(operation):
  """Takes in operation name and returns meta information.
  An example would be 'separable_3x3_4' -> (3, 4).
  Args:
    operation: String that corresponds to convolution operation.
  Returns:
    Tuple of (filter shape, num layers).
  """
  num_layers = _operation_to_num_layers(operation)
  filter_shape = _operation_to_filter_shape(operation)
  return num_layers, filter_shape


def _operation_to_pooling_type(operation):
  """Takes in the operation string and returns the pooling type."""
  splitted_operation = operation.split('_')
  return splitted_operation[0]


def _operation_to_pooling_shape(operation):
  """Takes in the operation string and returns the pooling kernel shape."""
  splitted_operation = operation.split('_')
  shape = splitted_operation[-1]
  assert 'x' in shape
  filter_height, filter_width = shape.split('x')
  assert filter_height == filter_width
  return int(filter_height)


def _operation_to_pooling_info(operation):
  """Parses the pooling operation string to return its type and shape."""
  pooling_type = _operation_to_pooling_type(operation)
  pooling_shape = _operation_to_pooling_shape(operation)
  return pooling_type, pooling_shape


"""Figure out what layers should have reductions."""
def calc_reduction_layers(num_cells, num_reduction_layers):
  reduction_layers = []
  for pool_num in range(1, num_reduction_layers + 1):
    layer_num = (float(pool_num) / (num_reduction_layers + 1)) * num_cells
    layer_num = int(layer_num)
    reduction_layers.append(layer_num)
  return reduction_layers

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

class net(nn.Module):
  def __init__(self,input_shape,filter_size,hiddenstate_indices,operations
        ,stride,drop_path_keep_prob,cell_num,total_training_steps
        ,total_num_cells,used_hiddenstates,prev_output = None):
    super().__init__()
    self.p1 =
    [
    nn.ReLU(),
    nn.Conv2d(in_channels=input_shape[1],out_channels=self.filter_size,kernel_size=1),
    nn.BatchNorm2d(self.filter_size)
    ]
    self.p2 = []
    self.p3 = []
    self.p4 = []
    self.p5 = []
    self.filter_size = filter_size
    self.prev_output = prev_output
    self.hiddenstate_indices=hiddenstate_indices
    self.operations = operations
    self.used_hiddenstates = used_hiddenstates


    if !(self.prev_output is None):
        if curr_filter_shape != prev_filter_shape:
            self.p2.append(nn.ReLU())
            layers = factorized_reduction(prev_output,self.filter_size, stride=2)
            if len(layers) == 1:
                for layer in layers[0]:
                    self.p2.append(layer)
            else:
                for layer in layers[0]:
                    self.p3.append(layer)
                for layer in layers[1]:
                    self.p4.append(layer)
                self.BN = nn.BatchNorm2d(self.filter_size)

        elif self.filter_size != prev_output.shape[1]:
            self.p2.append(nn.ReLU())
            self.p2.append(nn.Conv2d(in_channels=input_shape[1], out_channels=self.filter_size, kernel_size=1))
            self.p2.append(nn.BatchNorm2d(self.filter_size))

    i = 0
    for iteration in range(5):
      left_hiddenstate_idx = self.hiddenstate_indices[i]
      right_hiddenstate_idx = self.hiddenstate_indices[i + 1]

      original_input_left = left_hiddenstate_idx < 2
      original_input_right = right_hiddenstate_idx < 2

      operation_left = self.operations[i]
      operation_right = self.operations[i+1]
      i += 2
      self.p5 = self.p5 +
      [
      opera(input_shape,filter_size, operation_left ,stride, original_input_left
          ,drop_path_keep_prob,cell_num,total_training_steps,total_num_cells),
      opera(input_shape,filter_size, operation_right ,stride, original_input_right
          ,drop_path_keep_prob,cell_num,total_training_steps,total_num_cells)
      ]

  def forward(self, x):
      output1 = x
      output2 = x
      #calculate output1
      for layer in p1:
          output1 = layer(output1)
      #calculate output2
      if !(self.prev_output is None):
          if curr_filter_shape != prev_filter_shape:
              output2 = self.p2[0](output2)

              if len(layers) == 1:
                  for layer in self.p2[1:]:
                      output2 = layer(output2)

              else:
                  output3 = output2
                  output4 = output2

                  for layer in self.p3:
                      output3 = layer(output3)

                  for idx,layer in enumerate(self.p4):
                      if idx == 0:
                          output4 = layer(output4)[:,:,1:,1:]
                      else:
                          output4 = layer(output4)

                  output2 = torch.cat((output3, output4), dim=1)
                  output2 = self.BN(output2)

          elif self.filter_size != prev_output.shape[1]:
              for layer in self.p2:
                  output2 = layer(output2)

      output1 = torch.split(output1 , output1.shape[1] ,dim=1)
      output1 = list(output1)
      output1.append(output2)

      i = 0
      for iteration in range(5):
        left_hiddenstate_idx = self.hiddenstate_indices[i]
        right_hiddenstate_idx = self.hiddenstate_indices[i + 1]
        input1 = output1[left_hiddenstate_idx]
        input2 = output1[right_hiddenstate_idx]
        i += 2
        #to be continued  CHANNELS = self.filter_size
        h1 = self.p5[i-2](input1)
        h2 = self.p5[i-1](input2)
        h = h1 + h2
        output1.append(h)

      for layer in self.p5:
          layer.update_step()

      output1 = _combine_unused_states(output1,self.used_hiddenstates,)

      return output1



def _combine_unused_states(self, x , used_hiddenstates):
    """Concatenate the unused hidden states of the cell."""
    final_height = int(x[-1].shape[2])
    final_num_filters = int(x[-1].shape[1])
    assert len(used_hiddenstates) == len(x)

#WHY IS THIS FOR LOOP HERE IT CAN MAKE THINGS HARD
    for idx, used_h in enumerate(used_hiddenstates):
      curr_height = int(x[idx].shape[2])
      curr_num_filters = int(x[idx].shape[1])
      # Determine if a reduction should be applied to make the number of
      # filters match.
      should_reduce = final_num_filters != curr_num_filters
      should_reduce = (final_height != curr_height) or should_reduce
      should_reduce = should_reduce and not used_h
      if should_reduce:
        stride = 2 if final_height != curr_height else 1
        x[idx] = factorized_reduction(x[idx], final_num_filters, stride)

    states_to_combine = ([h for h, is_used in zip(x, used_hiddenstates) if not is_used])
    # Return the concat of all the states
    x = torch.cat(states_to_combine, dim=1)
    return x



def _stacked_separable_conv(stride, operation, filter_size):
  """Takes in an operations and parses it to the correct sep operation."""
  layers = []
  num_layers, kernel_size = _operation_to_info(operation)
  for layer_num in range(num_layers - 1):
    layers.append(nn.ReLU())
    layers.append(depthwise_separable_conv(filter_size,filter_size,kernel_size,depth_multiplier=1,stride=stride))
    layers.append(nn.BatchNorm2d(filter_size))
    stride = 1

  layers.append(nn.ReLU())
  layers.append(depthwise_separable_conv(filter_size,filter_size,kernel_size,depth_multiplier=1,stride=stride))
  layers.append(nn.BatchNorm2d(filter_size))
  return layers


class depthwise_separable_conv(nn.Module):
    def __init__(self, nin,nout,kernel_size,depth_multiplier,stride):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin * depth_multiplier, kernel_size=kernel_size, stride=stride, groups=nin)
        self.pointwise = nn.Conv2d(nin * depth_multiplier, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out



def _apply_conv_operation(self,operation, stride, filter_size):
    layers = []
    if operation == '1x1':
      layers.append(nn.Conv2d(in_channels=filter_size, out_channels=filter_size, kernel_size=1))
    elif operation == '3x3':
      layers.append(nn.Conv2d(in_channels=filter_size, out_channels=filter_size, kernel_size=3,stride=stride))
    elif operation == '1x7_7x1':
      layers.append(nn.Conv2d(in_channels=filter_size, out_channels=filter_size, kernel_size=[1,7],stride=[1,stride]))
      layers.append(nn.BatchNorm2d(filter_size))
      layers.append(nn.ReLU())
      layers.append(nn.Conv2d(in_channels=filter_size, out_channels=filter_size, kernel_size=[7,1],stride=[stride,1]))
    elif operation == '1x3_3x1':
      layers.append(nn.Conv2d(in_channels=filter_size, out_channels=filter_size, kernel_size=[1,3],stride=[1,stride]))
      layers.append(nn.BatchNorm2d(filter_size))
      layers.append(nn.ReLU())
      layers.append(nn.Conv2d(in_channels=filter_size, out_channels=filter_size, kernel_size=[3,1],stride=[stride,1]))
    elif operation in ['dilated_3x3_rate_2', 'dilated_3x3_rate_4','dilated_3x3_rate_6']:
      dilation_rate = int(operation.split('_')[-1])
      layers.append(nn.Conv2d(in_channels=filter_size, out_channels=filter_size, kernel_size=3,dilation=dilation_rate,stride=stride))
    else:
      raise NotImplementedError('Unimplemented conv operation: ', operation)
    return layers


def _pooling(net, stride, operation):
  """Parses operation and performs the correct pooling operation on net."""
  layer = None
  pooling_type, pooling_shape = _operation_to_pooling_info(operation)

  if pooling_type == 'avg':
    layer = nn.AvgPool2d(kernel_size=pooling_shape, stride=stride,padding=(pooling_shape - stride)//2,ceil_mode=True)
  elif pooling_type == 'max':
    layer = nn.MaxPool2d(kernel_size=pooling_shape, stride=stride,padding=(pooling_shape - stride)//2,ceil_mode=True)
  elif pooling_type == 'min':
    layer = nn.MaxPool2d(kernel_size=pooling_shape, stride=stride,padding=(pooling_shape - stride)//2,ceil_mode=True)
  else:
    raise NotImplementedError('Unimplemented pooling type: ', pooling_type)

  return layer,pooling_type


class opera(nn.Module):
    def __init__(self, input_shape,filter_size, operation,stride, is_from_original_input
        ,drop_path_keep_prob,cell_num,drop_path_burn_in_steps,num_cells):
      super().__init__()
      self.operation = operation
      self.layers = []
      self.stride = stride
      self.drop_path_keep_prob = drop_path_keep_prob
      self.cell_num = cell_num
      self.drop_path_burn_in_steps = drop_path_burn_in_steps
      self.num_cells = num_cells
      self.current_step = 0

      if stride > 1 and not is_from_original_input:
        stride = 1
      input_filters = input_shape[1]

      if 'separable' in operation:
        self.layers = _stacked_separable_conv(stride, operation, filter_size)

      elif operation in ['dilated_3x3_rate_2', 'dilated_3x3_rate_4',
                         'dilated_3x3_rate_6', '3x3', '1x7_7x1', '1x3_3x1']:
        if operation == '1x3_3x1':
          reduced_filter_size = int(3 * filter_size / 8)
        else:
          reduced_filter_size = int(filter_size / 4)

        if reduced_filter_size < 1:
          # If the intermediate number of channels would be too small, then don't
          # use a bottleneck layer.
          self.layers.append(nn.ReLU())
          self.layers = self.layers + _apply_conv_operation(operation, stride, filter_size)
          self.layers.append(nn.BatchNorm2d(filter_size))
        else:
          # Use a bottleneck layer.
          self.layers.append(nn.ReLU())
          self.layers.append(nn.Conv2d(in_channels=filter_size, out_channels=reduced_filter_size, kernel_size=1))
          self.layers.append(nn.BatchNorm2d(reduced_filter_size))
          self.layers.append(nn.ReLU())
          self.layers = self.layers + _apply_conv_operation(operation, stride, reduced_filter_size)
          self.layers.append(nn.BatchNorm2d(reduced_filter_size))
          self.layers.append(nn.ReLU())
          self.layers.append(nn.Conv2d(in_channels=reduced_filter_size, out_channels=filter_size, kernel_size=1))
          self.layers.append(nn.BatchNorm2d(filter_size))

      elif operation in ['none', '1x1']:
        # Check if a stride is needed, then use a strided 1x1 here
        if stride > 1 or operation == '1x1' or (input_filters != filter_size):
          self.layers.append(nn.ReLU())
          self.layers.append(nn.Conv2d(in_channels=filter_size, out_channels=filter_size, kernel_size=1,stride=stride))
          self.layers.append(nn.BatchNorm2d(filter_size))
      elif 'pool' in operation:
        l , _ = _pooling(stride, operation)
        self.layers.append(l)
        if input_filters != filter_size: #maybe channel dim is not right check later
          self.layers.append(nn.Conv2d(in_channels=filter_size, out_channels=filter_size, kernel_size=1,stride=1))
          self.layers.append(nn.BatchNorm2d(filter_size))
      else:
        raise ValueError('Unimplemented operation', operation)



    def forward(self,x):
        if 'separable' in self.operation or self.operation in ['dilated_3x3_rate_2', 'dilated_3x3_rate_4',
                           'dilated_3x3_rate_6', '3x3', '1x7_7x1', '1x3_3x1'] or self.operation in ['none', '1x1']:
            for layer in self.layers:
                x = layer(x)
        elif 'pool' in self.operation:
            _ , op = _pooling(stride, operation)
            if op == 'min':
                x = self.layers[0](-1*x)
                x = -1 * x
                for layer in self.layers[1:]:
                    x = layer(x)
            else:
                for layer in self.layers:
                    x = layer(x)
        if self.operation != 'none':
          x = _apply_drop_path(x,self.drop_path_keep_prob,self.cell_num,self.drop_path_burn_in_steps,self.num_cells,self.current_step)
        return x


    def update_step(self):
        self.current_step = current_step + 1



def _apply_drop_path(x,drop_path_keep_prob,cell_num,drop_path_burn_in_steps,num_cells,current_step,drop_connect_version='v1'):
    if drop_path_keep_prob < 1.0:
      assert drop_connect_version in ['v1', 'v2', 'v3']
      if drop_connect_version in ['v2', 'v3']:
        # Scale keep prob by layer number
        assert cell_num != -1
        # The added 2 is for the reduction cells
        layer_ratio = (cell_num + 1)/float(num_cells)
        drop_path_keep_prob = 1 - layer_ratio * (1 - drop_path_keep_prob)
      if drop_connect_version in ['v1', 'v3']:
        # Decrease the keep probability over time
        current_ratio = current_step / drop_path_burn_in_steps
        current_ratio = tf.minimum(1.0, current_ratio)
        drop_path_keep_prob = (1 - current_ratio * (1 - drop_path_keep_prob))
      x = drop_path(x, drop_path_keep_prob)
    return x


def drop_path(x, keep_prob, is_training=True):
  """Drops out a whole example hiddenstate with the specified probability."""
  if is_training:
    batch_size = x.shape[0]
    noise_shape = [batch_size, 1, 1, 1]
    keep_prob = keep_prob.type(x.type())
    random_tensor = keep_prob
    distribution = uniform.Uniform(torch.Tensor([0.0]),torch.Tensor([1.0]))
    random_tensor += distribution.sample(torch.Size(noise_shape))
    binary_tensor = torch.floor(random_tensor)
    x = torch.div(x,keep_prob) * binary_tensor
  return x


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
  def __call__(self, input ,prev_output=None,filter_scaling=1, stride=1,, cell_num=-1):
    """Runs the conv cell."""
    self._cell_num = cell_num
    self._filter_scaling = filter_scaling
    self._filter_size = int(self._num_conv_filters * filter_scaling)

    return net(input.shape,self._filter_size,self._hiddenstate_indices,self._operations,
            stride,self._drop_path_keep_prob,self._cell_num,self._total_training_steps,
            self._total_num_cells,self._used_hiddenstates,prev_output)
