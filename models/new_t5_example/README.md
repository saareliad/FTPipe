# Convert HuggingFace model to FTPipe format.

Download the model
```sh
wget https://raw.githubusercontent.com/huggingface/transformers/v4.5.1/src/transformers/models/t5/modeling_t5.py
```

Add
```python
from transformers.models.t5.modeling_t5 import *
```
to the top of the file to fix imports.

Run the conversion scrip to convert `is None` and `is not None`.

```python
from autopipe.autopipe.utils import convert_none_checks
convert_none_checks(input_file="modeling_t5.py", output_file="modeling_t5.py")
```

Use "stateless" addons to allow shared linear weights.

add
```python
from models.normal.NLP_models.stateless import StatelessEmbedding
```
to the top of the file.

Then, add
```
    def make_stateless(self):
        stateless_shared = StatelessEmbedding(self.shared)
        self.encoder.embed_tokens = StatelessEmbedding(self.shared)
        self.decoder.embed_tokens = StatelessEmbedding(self.shared)

        del self.shared
        self.encoder.embed_tokens.pop_weight()
        self.decoder.embed_tokens.pop_weight()

        self.shared_embed_weight = stateless_shared.pop_weight()
```
to T5Model.

This makes calls to `self.encoder.embed_tokens` and `self.decoder.embed_tokens` accept a the shared weight as the first parameter.

Then, make sure all calls get the new parameter  `self.shared_embed_weight`.
This requires the following changes in `forward` methods:

In `T5Stack`:

(1) Before:
```python
            inputs_embeds = self.embed_tokens(input_ids)
```

(1) After:
```python
            inputs_embeds = self.embed_tokens(shared_embedding, input_ids)
```

(2) add `shared_embedding` as first parameter in forward declaration.


(3) In `T5Modle`:
In callers:
`self.decoder(...)`
`self.encoder(...)`
simply add `self.shared_embed_weight` as the first parameter.

Now, the model can be registered to the framework. 

In addition: 
 - remove huggingface functions which are called in runtime but I'm too lazy to convert, like head mask (to remove `operator.is_`).
 - return a single value
 - check if there additional hidden `operator.is_`
 - `training=self.training` this is traced as static, replace it.

Explanation: 
1. Conversion: done to help the tracer. 
2. Stateless: this manually creates an edge from the shared weight to new `Staleless` layers, which will accept it as a parameter.
The rest will be handled by the framework.
3. single value: currently only models with single output value are supported (this can be easily changed).




