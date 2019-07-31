import amoebaNet_utils as au
import torch.nn as nn
import torch

class AmoebaNet(nn.Module):
  def __init__(self,inputs,num_classes,is_training,hparams=None):
      self.inputs = inputs
      self.num_classes = num_classes
      self.is_training = is_training
      self.hparams = hparams

      self.normal_cell_factory = au.BaseCell(
      hparams.reduction_size,
      hparams.normal_cell_operations,
      hparams.normal_cell_used_hiddenstates,
      hparams.normal_cell_hiddenstate_indices,
      hparams.drop_connect_keep_prob,
      hparams.num_cells + 4,
      hparams.num_total_steps
      )

      self.reduction_cell_factory = au.BaseCell(
      hparams.reduction_size,
      hparams.reduction_cell_operations,
      hparams.reduction_cell_used_hiddenstates,
      hparams.reduction_cell_hiddenstate_indices,
      hparams.drop_connect_keep_prob,
      hparams.num_cells + 4,
      hparams.num_total_steps
      )
      self.image_stem = imagenet_stem(inputs, hparams, self.reduction_cell_factory, 2)
      cell_outputs = self.image_stem.get_shape()
      self.red = []
      self.norm = []
      self.aux = []
      # Run the cells
      filter_scaling = 1.0
      # true_cell_num accounts for the stem cells
      true_cell_num = 2
      reduction_indices = au.calc_reduction_layers(self.hparams.num_cells, self.hparams.num_reduction_layers)
      aux_head_cell_idxes = []
      if len(reduction_indices) >= 2:
        aux_head_cell_idxes.append(reduction_indices[1] - 1)

      for cell_num in range(hparams.num_cells):
        stride = 1

        if cell_num in reduction_indices:
          filter_scaling *= 2
          red_cell = self.reduction_cell_factory(cell_outputs[-1],prev_layer=cell_outputs[-2],filter_scaling=filter_scaling,stride=2,cell_num=true_cell_num)
          true_cell_num += 1
          cell_outputs.append(torch.randn(red_cell.get_shape()))
          self.red.append(red_cell)
        else:
          self.red.append([])

        norm_cell = self.normal_cell_factory(cell_outputs[-1],prev_layer=cell_outputs[-2],filter_scaling=filter_scaling,stride,true_cell_num)
        true_cell_num += 1
        self.norm.append(norm_cell)
        out_shape = normal_cell.get_shape()
        cell_outputs.append(torch.randn(out_shape))

        if (hparams.use_aux_head and cell_num in aux_head_cell_idxes and num_classes and is_training):
          self.aux.append(nn.ReLU())
          self.aux.append(aux_head(torch.randn(out_shape) , num_classes , hparams))

      self.layers =
      [
       nn.ReLU(),
       nn.Dropout(p=hparams.dense_dropout_keep_prob),
       nn.Linear(in_features=cell_outputs[-1].shape[1],out_features=num_classes)
      ]

      self.soft = nn.Softmax()





  def forward(self,x):
      end_points = {"aux_logits":[]}
      filter_scaling_rate = 2
      reduction_indices = au.calc_reduction_layers(self.hparams.num_cells, self.hparams.num_reduction_layers)
      aux_head_cell_idxes = []
      if len(reduction_indices) >= 2:
        aux_head_cell_idxes.append(reduction_indices[1] - 1)

      cell_outputs = self.image_stem(x)

      i = 0
      for cell_num in range(hparams.num_cells):
        stride = 1

        if cell_num in reduction_indices:
            output = self.red[cell_num](cell_outputs[-1])
            cell_outputs.append(output)

        output = self.norm[cell_num](cell_outputs[-1])

        to_aux = output
        to_list = output

        if (self.hparams.use_aux_head and cell_num in aux_head_cell_idxes and self.num_classes and self.is_training):
          to_aux = self.aux[i](to_aux)
          to_aux = self.aux[i+1](to_aux)
          to_aux = to_aux.type(torch.float32)
          end_points["aux_logits"].append(to_aux)
          i = i + 2

        cell_outputs.append(to_list)

      out = self.layers[0](cell_outputs[-1])
      out = torch.mean(out,[2,3])

      for layer in self.layers[1:]:
        out = layer(out)

      logit = out.type(torch.float32)
      predictions = self.soft(logit)
      end_points["logits"] = logit
      end_points["predictions"] = predictions
      return logit,end_points

class imagenet_stem(nn.Module):
  def __init__(inputs, hparams, stem_cell_factory: au.BaseCell , filter_scaling_rate):
      self.filter_scaling_rate = filter_scaling_rate
      num_stem_cells = 2
      # 32 x 149 x 149
      num_stem_filters = hparams.stem_reduction_size
      self.layers = []

      track_shape = inputs.shape
      self.conv = nn.Conv2d(inputs.shape[1], num_stem_filters, kernel_size=3,stride=2)
      self.BN = nn.BatchNorm2d(num_stem_filters)
      track_shape[1] = num_stem_filters
      track_shape[2] = int( ( (track_shape[2]-3)/2 )+1)
      track_shape[3] = int( ( (track_shape[3]-3)/2 )+1)

      filter_scaling = 1.0 / (filter_scaling_rate**num_stem_cells)

      self.cell_outputs = [None,torch.randn(track_shape)]
      for cell_num in range(num_stem_cells):
          stem_cell = stem_cell_factory(self.cell_outputs[-1],self.cell_outputs[-2],filter_scaling,2,cell_num)
          self.layers.append(stem_cell)
          self.cell_outputs.append(torch.randn(stem_cell.get_shape()))
          filter_scaling *= filter_scaling_rate


  def forward(self,x):
      x = self.conv(x)
      x = self.BN(x)
      num_stem_cells = 2
      cell_outputs = [None,x]
      filter_scaling = 1.0 / (self.filter_scaling_rate**num_stem_cells)
      for cell in self.layers:
          x = cell(x)
          cell_outputs.append(x)
          filter_scaling *= filter_scaling_rate
      return cell_outputs


  def get_shape(self,):
      return self.cell_outputs


class aux_head(nn.Module):
  def __init__(inputs, num_classes , hparams):
      self.layers = []
      track_shape = inputs.shape
      aux_scaling = 1.0
      if hasattr(hparams, 'aux_scaling'):
          aux_scaling = hparams.aux_scaling

      self.layers.append(nn.AvgPool2d(kernel_size=5,stride=3,padding=0,ceil_mode=True))
      track_shape[2] = ceil(((track_shape[2]-5)/3) + 1)
      track_shape[3] = ceil(((track_shape[3]-5)/3) + 1)
      self.layers.append(nn.Conv2d(track_shape[1],int(128 * aux_scaling),1))
      track_shape[1] = int(128 * aux_scaling)
      self.layers.append(nn.BatchNorm2d(track_shape[1]))
      self.layers.append(nn.ReLU())

      self.layers.append(nn.Conv2d(track_shape[1],int(768 * aux_scaling),kernel_size=track_shape[2:4],padding=0,ceil_mode=True))
      track_shape[1] = int(768 * aux_scaling)
      track_shape[2] = 1
      track_shape[3] = 1
      self.layers.append(nn.BatchNorm2d(track_shape[1]))
      self.layers.append(nn.ReLU())
      self.layers.append(Flatten())
      self.layer.append(nn.Linear(in_features=track_shape[1],out_features=num_classes))

  def forward(self,x):
      for layer in self.layers:
          x = layer(x)
      return x



class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
