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

