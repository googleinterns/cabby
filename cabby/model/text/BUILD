package(
    default_visibility = ["//visibility:public"],
)

py_library(
    name = 'input_pipeline',
    srcs = ['input_pipeline.py'],
)

py_library(
    name = 'model',
    srcs = ['model.py'],
)

py_library(
    name = 'dataset',
    srcs = ['dataset.py'],
)

py_binary(
    name = 'train',
    main = 'train.py',
    srcs = ['train.py'],
    deps = [
        ':input_pipeline',
        ':model',
    ],
)

py_binary(
    name = 'train_2',
    main = 'train_2.py',
    srcs = ['train_2.py'],
    deps = [
        ':dataset',
    ],
)