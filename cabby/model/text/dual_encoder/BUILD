# Cabby: Geographic vision-and-language navigation

licenses(["notice"])

exports_files(["LICENSE"])

package(
    default_visibility = ["//visibility:public"],
)



py_library(
    name = 'dataset_wikigeo',
    srcs = ['dataset_wikigeo.py'],
    deps = [
          ":dataset_item",
          "//cabby/geo:util",
          "//cabby/geo:regions",
          "//cabby/model:util",

    ],
)
py_library(
    name = 'dataset_rvs',
    srcs = ['dataset_rvs.py'],
    deps = [
          ":dataset_item",
          "//cabby/model:util",
          "//cabby/geo:util",
          "//cabby/geo:regions",
          "//cabby/model:datasets",


    ],
)

py_library(
    name = 'dataset_run',
    srcs = ['dataset_run.py'],
    deps = [
          ":dataset_item",
          "//cabby/model:util",
          "//cabby/geo:util",
          "//cabby/geo:regions",
          "//cabby/model:datasets",
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
          "//cabby/evals:utils",

    ],
)

py_binary(
    name = 'model_trainer',
    main = 'model_trainer.py',
    srcs = ['model_trainer.py'],
    deps = [
        ':train',
        ':dataset_wikigeo',
        ':dataset_rvs',
        ':dataset_run',
        ':dataset_item',
        ':model',
        "//cabby/model/text:util",

    ],
)