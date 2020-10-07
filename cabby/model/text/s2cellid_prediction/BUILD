package(
    default_visibility = ["//visibility:public"],
)


py_library(
    name = 'dataset',
    srcs = ['dataset.py'],
    deps = [
          "//cabby/geo:util",
          "//cabby/geo/map_processing:map_structure",

    ],
)

py_library(
    name = 'rvs_dataset',
    srcs = ['rvs_dataset.py'],
    deps = [
          "//cabby/geo:util",
          "//cabby/geo/map_processing:map_structure",

    ],
)

py_binary(
    name = 'train',
    main = 'train.py',
    srcs = ['train.py'],
    deps = [

    ],
)

py_binary(
    name = 'model_trainer',
    main = 'model_trainer.py',
    srcs = ['model_trainer.py'],
    deps = [
        ':train',
        ':dataset',
    ],
)


py_binary(
    name = 'rvs_evaluation',
    main = 'rvs_evaluation.py',
    srcs = ['rvs_evaluation.py'],
    deps = [
        ':train',
        ':rvs_dataset',
    ],
)