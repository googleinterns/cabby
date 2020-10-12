package(
    default_visibility = ["//visibility:public"],
)


py_library(
    name = 'dataset',
    srcs = ['dataset.py'],
    deps = [
          "//cabby/geo:util",
          "//cabby/geo:regions",

    ],
)

py_binary(
    name = 'train',
    main = 'train.py',
    srcs = ['train.py'],
    deps = [
          "//cabby/geo:util",
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