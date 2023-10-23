local transformer_model = std.extVar("ENCODER");
local onto_path = std.extVar("DATA");
local cuda_devices = std.parseJson(std.extVar("DEVICE"));
local archive_path = std.extVar("ARCHIVE");
local segment_length = std.parseJson(std.extVar("SEGMENT"));
#local max_tokens = std.parseJson(std.extVar("MAX_TOKENS"));
local max_tokens = 1200;

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
  model: {
    type: "from_archive",
    archive_file: archive_path,
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
