import logging
import os
import t5
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds
import torch
from transformers import T5ForConditionalGeneration
import functools
from .t5_squad import load_huggingface_checkpoint
get_dataset = t5.models.hf_model.get_dataset
from tqdm import tqdm

class T5Evaluator:
    def __init__(self, args, model_dir, device, model: T5ForConditionalGeneration = None):
        super().__init__()
        self._model: T5ForConditionalGeneration = None
        self._writer = torch.utils.tensorboard.writer.SummaryWriter(model_dir)
        self._model_dir = model_dir
        if isinstance(device, str):
            device = torch.device(device)
        self._device = device

        if model is not None:
            self._model = model
            if self._device.type == "cuda":
                self._model.to(device)

        # self._step = 0
        # self.load_latest_checkpoint()
        self.to_tensor = functools.partial(torch.as_tensor, device=self._device)
        self.args = args

    def load_checkpoint(self, cp_number):
        # TODO: can use existing model to save load time by passing
        # {
        #     "model": self._model
        #     "tokenizer": None
        #     "config":None
        # }
        # to `load_huggingface_checkpoint`, but decided not to do it because not sure about internal state HF saves.
        # so better load from scratch to be sure.
        hugg, tokenizer = load_huggingface_checkpoint(args=self.args, cp_number=cp_number)
        self._model = hugg
        self._step = cp_number  # HACK

        if self._device.type == "cuda":
            self._model.to(self._device)

    def get_all_checkpoint_steps(self):
        raise NotImplementedError()

    def eval(
        self,
        mixture_or_task_name,
        sequence_length,
        batch_size,
        checkpoint_steps=None,
        summary_dir=None,
        split="validation",
        **generate_kwargs,
    ):
        """Evaluate the model on the given Mixture or Task.

        *Note*: If a checkpoint step is provided (i.e. `checkpoint_steps is not
        None`), the model's state will be replaced by the state in those
        checkpoints. If you have not saved your model before calling `eval`, you
        should call `save_checkpoint` before `eval` to avoid losing its parameter
        values and state.

        Args:
            mixture_or_task_name: str, the name of the Mixture or Task to evaluate
            on.  Must be pre-registered in the global `t5.data.TaskRegistry` or
            `t5.data.MixtureRegistry.`
            sequence_length: dict of int, a dict mapping feature name to length.
            batch_size: int, the number of padded sequences in each batch.
            checkpoint_steps: int, list of ints, "all", or None. If None, eval in the
            model in its current state without loading any checkpoints. If an int
            or list of ints, evaluation will be run on the checkpoint files in
            `model_dir` whose global steps are those provided. If -1, eval on the
            latest checkpoint from the model directory. If "all", evaluate all
            checkpoints in the model directory.
            summary_dir: str, path to write TensorBoard events file summaries for
            eval. If None, use model_dir/{split}_eval.
            split: str, the mixture/task split to evaluate on.
            **generate_kwargs: Additional keyword arguments to pass to
            `transformers.PretrainedModel.generate()`, for example to change the
            decoding strategy. See the documentation for
            `transformers.PretrainedModel.generate()` for options.
        """
        mixture_or_task = t5.data.get_mixture_or_task(mixture_or_task_name)
        vocab = mixture_or_task.output_features["targets"].vocabulary

        if isinstance(mixture_or_task, t5.data.Mixture):
            tasks = mixture_or_task.tasks
        elif isinstance(mixture_or_task, t5.data.Task):
            tasks = [mixture_or_task]
        else:
            raise NotImplementedError()

        for task in tasks:
            if split not in task.splits:
                logging.info(
                    "Task %s has no '%s' split; skipping eval.", task.name, split
                )
        tasks = [task for task in tasks if split in task.splits]

        summary_dir = summary_dir or os.path.join(self._model_dir, f"{split}_eval")
        tf.io.gfile.makedirs(summary_dir)

        def _unbatch(batch):
            """Converts a dict of lists to a list of dicts of singletons."""
            return [dict(zip(batch, t)) for t in zip(*batch.values())]

        # Pre-load in all of the targets once before doing eval
        cached_targets = {}
        cached_examples = {}
        for task in tasks:
            if task.metric_fns:
                ds = get_dataset(task.name, sequence_length, split, batch_size)
                # Create list of postprocessed text targets
                batches = list(ds)
                if not batches:
                    raise ValueError(f"The '{split}' split of {task.name} is empty.")
                # "Unbatch" the dataset
                examples = [ex for b in batches for ex in _unbatch(b)]  # pylint:disable=g-complex-comprehension
                targets = [
                    task.postprocess_fn(  # pylint:disable=g-complex-comprehension
                        tf.compat.as_text(ex["targets_plaintext"]),
                        example=ex,
                        is_target=True
                    ) for ex in examples
                ]
                targets_filename = os.path.join(summary_dir, f"{task.name}_targets")
                write_lines_to_file(targets, targets_filename)

                inputs_filename = os.path.join(summary_dir, f"{task.name}_inputs")
                inputs = [ex["inputs_plaintext"] for ex in examples]
                write_lines_to_file(inputs, inputs_filename)

                cached_targets[task.name] = targets
                cached_examples[task.name] = batches

        def _eval_current_model():
            self._model.eval()
            all_scores = {}

            for task in tasks:
                ds = cached_examples[task.name]
                targets = cached_targets[task.name]
                predictions = []
                # TODO: attention_mask ?
                for batch in tqdm(ds, desc="Evaluating"):
                    predicted_tokens = self._model.generate(
                        input_ids=self.to_tensor(batch["inputs"]), **generate_kwargs
                    )
                    predicted_tokens = predicted_tokens.cpu().numpy().tolist()
                    predictions.extend(
                        [
                            task.postprocess_fn(vocab.decode(p), example=ex)
                            for p, ex in zip(predicted_tokens, _unbatch(batch))
                        ]
                    )

                if len(targets) != len(predictions):
                    raise ValueError(
                        f"#targets ({len(targets)}) != #predictions ({len(predictions)})"
                    )

                predictions_file = os.path.join(
                    summary_dir, f"{task.name}_{self._step}_predictions"
                )
                write_lines_to_file(predictions, predictions_file)


                for metric_fn in task.metric_fns:
                    scores = metric_fn(targets, predictions)
                    for metric_name, metric_value in scores.items():
                        tag = f"eval/{task.name}/{metric_name}"
                        step = self._step if isinstance(self._step, int) else -1
                        self._writer.add_scalar(tag, metric_value, step)
                        logging.info(
                            f"{tag} at step {step}: {metric_value:.3f}"
                        )
                        all_scores[tag] = metric_value

                self._writer.flush()

            return all_scores

        if checkpoint_steps is None:
            raise NotImplementedError()
            # _eval_current_model()
            # return
        elif isinstance(checkpoint_steps, int):
            checkpoint_steps = [checkpoint_steps]
        elif checkpoint_steps == "all":
            checkpoint_steps = self.get_all_checkpoint_steps()
        elif not isinstance(checkpoint_steps, (list, tuple)):
            raise ValueError(
                f"checkpoint_steps must be None, int or list; got {checkpoint_steps}"
            )

        all_results = {}
        for checkpoint_step in checkpoint_steps:
            self.load_checkpoint(checkpoint_step)
            results = _eval_current_model()
            print("partial result:", checkpoint_step, results)
            all_results[checkpoint_step] = results

        return all_results


def write_lines_to_file(lines, filename):
    """Write each line to filename, replacing the file if it exists."""
    if tf.io.gfile.exists(filename):
        tf.io.gfile.remove(filename)
    with tf.io.gfile.GFile(filename, "w") as output_file:
        output_file.write("\n".join([str(l) for l in lines]))