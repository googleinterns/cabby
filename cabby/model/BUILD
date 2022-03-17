# Cabby: Geographic vision-and-language navigation

licenses(["notice"])

exports_files(["LICENSE"])

package(
    default_visibility = ["//visibility:public"],
)

py_library(
    name = 'util',
    srcs = ['util.py'],    
    deps = [
        "//cabby/geo:util",

],
)


py_binary(
    name = 'baselines',
    main = 'baselines.py',
    srcs = ['baselines.py'],
    deps = [
          ":datasets",
          "//cabby/geo:util",
          "//cabby/model:util",
          "//cabby/geo:regions",
          "//cabby/evals:utils"

    ],
)


py_binary(
    name = 'datasets',
    main = 'datasets.py',
    srcs = ['datasets.py'],
    deps = [
          "//cabby/geo:util",
          "//cabby/geo:regions",
          "//cabby/evals:utils",
          ":dataset_item",
          ":util"


    ],
)

py_library(
    name = 'dataset_item',
    srcs = ['dataset_item.py'],
    deps = [
        "//cabby/geo:util",
        ":util"


],

)



py_test(
    name = "util_test",
    srcs = ["util_test.py"],
    python_version = "PY3",
    deps = [":util"],
)
