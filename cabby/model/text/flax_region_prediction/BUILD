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


py_binary(
    name = 'train',
    main = 'train.py',
    srcs = ['train.py'],
    deps = [
        ':input_pipeline',
        ':model',
    ],
)