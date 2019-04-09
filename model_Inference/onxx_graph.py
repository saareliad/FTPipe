import torch
# import models
import torch.jit

import torch.nn as nn
if __name__ == "__main__":

    dummy_input = torch.randn(1, 3, 32, 32, device='cuda')
    # model = models.GoogLeNet().cuda()

    # # Providing input and output names sets the display names for values
    # # within the model's graph. Setting these does not change the semantics
    # # of the graph; it is only for readability.
    # #
    # # The inputs to the network consist of the flat list of inputs (i.e.
    # # the values you would pass to the forward() method) followed by the
    # # flat list of parameters. You can partially specify names, i.e. provide
    # # a list here shorter than the number of inputs to the model, and we will
    # # only set that subset of names, starting from the beginning.
    # input_names = ["actual_input_1"] + ["learned_%d" % i for i in range(16)]
    # output_names = ["output1"]

    # st = torch.onnx.export_to_pretty_string(model, dummy_input, f="abc.txt", verbose=False,
    #                                         input_names=input_names, output_names=output_names)

    # graph, params, torch_out = utils._model_to_graph(model, dummy_input, "", verbose=False, training=False,
    #                                                  input_names=input_names, output_names=output_names,
    #                                                  example_outputs=None, propagate=False)

    # # pprint(dir(graph))

    # print(graph.at(0))

    # produce trace graph

    def trace_graph(network, x):
        trace, out = torch.jit.get_trace_graph(network, x)

        torch.onnx._optimize_trace(
            trace, torch.onnx.OperatorExportTypes.ONNX)

        return trace.graph()

    # print(pprint(vars(torch.onnx.OperatorExportTypes)))
    # print(torch_graph)
    # print(torch_graph.at("12"))

    # 'ONNX': OperatorExportTypes.ONNX,
    # 'ONNX_ATEN': OperatorExportTypes.ONNX_ATEN,
    # 'ONNX_ATEN_FALLBACK': OperatorExportTypes.ONNX_ATEN_FALLBACK,
    # 'RAW': OperatorExportTypes.RAW,

    # print(len(list(model.named_buffers())))
    # print(len(list(model.parameters())))

    # print("modules")
    # print(len(list(model.modules())))
    # for module in model.modules():
    #     print(module)

    # print("\nchildren")
    # print(len(list(model.children())))
    # for child in model.children():
    #     print(child)

    # buffers = list(model.named_buffers())
    # parameters = list(model.named_parameters())

    # layers = list(model.children())

    # print(len(layers))

    # print(layers)

    # print(f"parameters: {len(parameters)}")
    # for name, p in parameters:
    #     print(f"{name}\n")

    # print(torch_graph)

    class complexNet(nn.Module):
        def __init__(self):
            super(complexNet, self).__init__()
            a = nn.Linear(2, 2)

            self.sub1 = nn.Sequential(
                nn.Sequential(a),
                a, nn.Linear(2, 2), nn.Sequential(nn.Linear(2, 2)))

            self.sub2 = nn.Linear(2, 1)

        def forward(self, x):
            return self.sub2(self.sub1(x))

    # get all individual layers

    def list_layers_no_prefix(network: nn.Module):
        l = []
        for m in network.modules():
            if len(list(m.children())) == 0:
                l.append(m)

        return l

    # create flatten dict for layers

    def layers_dict(network: nn.Module):
        layer_dict = {}
        for name, m in network.named_modules():
            # add only leaf modules
            if len(list(m.children())) == 0:
                layer_dict[name] = m

        return layer_dict

    model = complexNet()
