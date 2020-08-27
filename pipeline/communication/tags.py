from models.simple_partitioning_config import PipelineConfig


# TODO tags for tensors we send multiple times
def tensor_tags_from_config(pipe_config: PipelineConfig, num_chunks=1, target_tensor_names=None, GRAD_UGLY_SHAMEFUL_NAME="_grad"):
    # Note: same tags for all process

    tensor_tags = {}
    tensor_tag = 1

    for i, stage in pipe_config.d['stages'].items():
        input_tensors = stage['inputs']
        output_tensors = stage['outputs']
        req_grad = pipe_config.get_inputs_req_grad_for_stage(i)
        outputs_req_grad = pipe_config.get_outputs_req_grad_for_stage(i)
        # Create different tags for gradients
        for name_post_addition in ["", GRAD_UGLY_SHAMEFUL_NAME]:
            for input_tensor in input_tensors:
                if input_tensor not in req_grad:
                    assert input_tensor in pipe_config.d['model_inputs']
                    continue
                input_tensor += name_post_addition
                if input_tensor not in tensor_tags:
                    tensor_tags[input_tensor] = tensor_tag
                    tensor_tag += num_chunks

            for output_tensor in output_tensors:
                if output_tensor not in outputs_req_grad:
                    assert output_tensor in pipe_config.d['model_outputs']
                    continue
                output_tensor += name_post_addition
                if output_tensor not in tensor_tags:
                    tensor_tags[output_tensor] = tensor_tag
                    tensor_tag += num_chunks

    if target_tensor_names:
        for target_tensor_name in sorted(target_tensor_names):
            tensor_tags[target_tensor_name] = tensor_tag
            tensor_tag += num_chunks

    total_tags = len(tensor_tags)
    return tensor_tags, total_tags