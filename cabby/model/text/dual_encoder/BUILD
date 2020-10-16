# Cabby: Geographic vision-and-language navigation

licenses(["notice"])

exports_files(["LICENSE"])

package(
    default_visibility = ["//visibility:public"],
)



py_library(
    name = 'dataset',
    srcs = ['dataset.py'],
    deps = [
          ":dataset_item",
          "//cabby/geo:util",
          "//cabby/geo:regions",

    ],
)


py_library(
    name = 'dataset_item',
    srcs = ['dataset_item.py'],

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
          "//cabby/model/text:util",

    ],
)

py_binary(
    name = 'model_trainer',
    main = 'model_trainer.py',
    srcs = ['model_trainer.py'],
    deps = [
        ':train',
        ':dataset',
        ':dataset_item',
        ':model',
        "//cabby/model/text:util",

    ],
)