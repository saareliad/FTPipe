from .modeling_bert import (BertPreTrainedModel, BertModel, BertForPreTraining,
                            BertForMaskedLM, BertForNextSentencePrediction,
                            BertForSequenceClassification, BertForMultipleChoice,
                            BertForTokenClassification, BertForQuestionAnswering,
                            load_tf_weights_in_bert, BERT_PRETRAINED_MODEL_ARCHIVE_MAP)

from .modeling_gpt2 import (GPT2PreTrainedModel, GPT2Model,
                            GPT2LMHeadModel, GPT2DoubleHeadsModel,
                            load_tf_weights_in_gpt2, GPT2_PRETRAINED_MODEL_ARCHIVE_MAP)
