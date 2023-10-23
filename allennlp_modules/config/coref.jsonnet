local transformer_model = std.extVar("ENCODER");
local onto_path = std.extVar("DATA");
local segment_length = std.parseJson(std.extVar("SEGMENT"));
local use_global = std.parseJson(std.extVar("USE_GLOBAL"));
local mem_length = std.parseJson(std.extVar("MEM"));
local cuda_devices = std.parseJson(std.extVar("DEVICE"));
local max_tokens = std.parseJson(std.extVar("MAX_TOKENS"));
local no_load = false;
local use_performer = std.parseJson(std.extVar("PERFORMER"));
local overlap = std.parseJson(std.extVar("OVERLAP"));
local lstm = std.parseJson(std.extVar("LSTM"));

local feature_size = 20;
local max_span_width = 30;
local debug = false;

{
  "dataset_reader": {
    "type": "conll",
    "token_indexers": {
      "tokens": {
        "type": "pretrained_transformer_mismatched",
        "model_name": transformer_model,
        "max_length": segment_length,
      },
    },
    "max_span_width": max_span_width,
    "max_sentences": 100,
    max_tokens: max_tokens,
    [if debug then "max_instances"]: 16,
  },
  "validation_dataset_reader": {
    "type": "coref",
    "token_indexers": {
      "tokens": {
        "type": "pretrained_transformer_mismatched",
        "model_name": transformer_model,
        "max_length": segment_length,
      },
    },
    "max_span_width": max_span_width,
    [if debug then "max_instances"]: 16,
  },
  "train_data_path": onto_path + '/train.english.v4_gold_conll',
  "validation_data_path": onto_path + '/dev.english.v4_gold_conll',
  "test_data_path": onto_path + '/test.english.v4_gold_conll',
  "model": {
    "type": "coarse2fine",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
            "type": "lre_mismatched",
            gradient_checkpointing: true,
            "model_name": transformer_model,
            "mem_length": mem_length,
            "max_length": segment_length,
            "use_global": use_global,
            [if no_load then "load_weights"]: false,
            [if use_performer then "attn_args"]: {type: "performer"},
            "lstm": lstm,
            "overlapping": overlap,
        }
      }
    },
    "context_layer": {
        "type": "pass_through",
    },
    "mention_feedforward": {
        "num_layers": 2,
        "hidden_dims": 1500,
        "activations": "relu",
        "dropout": 0.3
    },
    "antecedent_feedforward": {
        "num_layers": 2,
        "hidden_dims": 1500,
        "activations": "relu",
        "dropout": 0.3
    },
    "initializer": {
      "regexes": [
        [".*_span_updating_gated_sum.*weight", {"type": "xavier_normal"}],
        [".*linear_layers.*weight", {"type": "xavier_normal"}],
        [".*scorer.*weight", {"type": "xavier_normal"}],
        ["_distance_embedding.weight", {"type": "xavier_normal"}],
        ["_span_width_embedding.weight", {"type": "xavier_normal"}],
        ["_context_layer._module.weight_ih.*", {"type": "xavier_normal"}],
        ["_context_layer._module.weight_hh.*", {"type": "orthogonal"}]
      ]
    },
    "feature_size": feature_size,
    "max_span_width": max_span_width,
    "spans_per_word": 0.4,
    "max_antecedents": 50,
    "coarse_to_fine": true,
    "inference_order": 2
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      # Explicitly specifying sorting keys since the guessing heuristic could get it wrong
      # as we a span field.
      "sorting_keys": ["text"],
      "batch_size": 1
    }
  },
  "trainer": {
    "num_epochs": 40,
    "patience" : 10,
    "validation_metric": "+coref_f1",
    [if std.isArray(cuda_devices) && std.length(cuda_devices) == 1 then "cuda_device"]: cuda_devices[0],
    [if std.isNumber(cuda_devices) then "cuda_device"]: cuda_devices,
    "learning_rate_scheduler": {
      "type": "slanted_triangular",
      "cut_frac": 0.06
    },
    "optimizer": {
      "type": "huggingface_adamw",
      "lr": 3e-4,
      "parameter_groups": [
        [[".*transformer.*"], {"lr": 1e-5}]
      ]
    },
    "checkpointer": {
      "type": "default",
      "keep_most_recent_by_count": 1,
    },

  },
  [if std.isArray(cuda_devices) && std.length(cuda_devices) > 1 then "distributed"]: {
    "cuda_devices": cuda_devices
  },
  "evaluate_on_test": true,
}
