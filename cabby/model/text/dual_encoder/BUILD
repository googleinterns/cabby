# Cabby: Geographic vision-and-language navigation

licenses(["notice"])

exports_files(["LICENSE"])

package(
    default_visibility = ["//visibility:public"],
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
          "//cabby/model:util",
          "//cabby/evals:utils",

    ],
)

py_binary(
    name = 'model_trainer',
    main = 'model_trainer.py',
    srcs = ['model_trainer.py'],
    deps = [
        ':train',
        '//cabby/model:datasets',
        '//cabby/model:dataset_item',
        ':model',
        "//cabby/model:util",

    ],
)