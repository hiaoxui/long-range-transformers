local transformer_model = std.extVar("ENCODER");
local data_path = std.extVar("DATA");
local segment_length = std.parseJson(std.extVar("SEGMENT"));
local use_global = std.parseJson(std.extVar("USE_GLOBAL"));
local mem_length = std.parseJson(std.extVar("MEM"));
local cuda_devices = std.parseJson(std.extVar("DEVICE"));
local max_total_length = std.parseJson(std.extVar("MAX_LENGTH"));
local batch_token = std.parseJson(std.extVar("BATCH_TOKEN"));
local use_performer = std.parseJson(std.extVar("PERFORMER"));
local no_load = std.parseJson(std.extVar("NO_LOAD"));
local max_in_memory = 2048;
local debug = false;
local train_split_size = 32768;
local dev_split_size = 1024;

{
  dataset_reader: {
    type: "docnli",
    pretrained_model: transformer_model,
    debug: debug,
    max_total_length: max_total_length,
    max_length: segment_length,
    [if debug then "max_instances"]: 16,
    split_size: train_split_size,
  },
  validation_dataset_reader: {
    type: "docnli",
    pretrained_model: transformer_model,
    debug: debug,
    max_total_length: max_total_length,
    max_length: segment_length,
    [if debug then "max_instances"]: 16,
    split_size: dev_split_size,
  },
  train_data_path: data_path + '/train.json',
  validation_data_path: data_path + '/dev.json',
  test_data_path: data_path + '/test.json',
  model: {
    type: "nli",
    text_field_embedder: {
      token_embedders: {
        tokens: {
            type: "lre",
            model_name: transformer_model,
            mem_length: mem_length,
            max_length: segment_length,
            use_global: use_global,
            first_global: true,
            [if no_load then "load_weights"]: false,
            [if use_performer then "attn_args"]: {type: "performer"},
        }
      }
    },
    dense: {
        num_layers: 1,
        hidden_dims: 768,
        activations: "tanh",
        dropout: 0.1
    },
  },
  datasets_for_vocab_creation: [],
  data_loader: {
    batch_sampler: {
      type: "max_tokens_sampler",
      max_tokens: batch_token,
      sorting_keys: ["tokens"],
    },
    max_instances_in_memory: max_in_memory,
  },
  validation_data_loader: {
    batch_sampler: {
      type: "max_tokens_sampler",
      max_tokens: batch_token*2,
      sorting_keys: ["tokens"],
    },
    max_instances_in_memory: max_in_memory,
  },
  trainer: {
    num_epochs: 29,
    patience : 29,
    # validation_metric: "+acc",
    [if std.isArray(cuda_devices) && std.length(cuda_devices) == 1 then "cuda_device"]: cuda_devices[0],
    [if std.isNumber(cuda_devices) then "cuda_device"]: cuda_devices,
    learning_rate_scheduler: {
      type: "slanted_triangular",
      cut_frac: 0.06
    },
    optimizer: {
      type: "huggingface_adamw",
      lr: 5e-5,
      parameter_groups: [
        [[".*transformer.*"], {"lr": 5e-6}]
      ]
    },
    checkpointer: {
      type: "default",
      keep_most_recent_by_count: 5,
    },
  },
  [if std.isArray(cuda_devices) && std.length(cuda_devices) > 1 then "distributed"]: {
    cuda_devices: cuda_devices
  },
  evaluate_on_test: true,
}
