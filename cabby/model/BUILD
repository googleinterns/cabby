# Cabby: Geographic vision-and-language navigation

licenses(["notice"])

exports_files(["LICENSE"])

package(
    default_visibility = ["//visibility:public"],
)

py_library(
    name = 'util',
    srcs = ['util.py'],    
)


py_binary(
    name = 'baselines',
    main = 'baselines.py',
    srcs = ['baselines.py'],
    deps = [
          ":datasets",
          "//cabby/geo:util",
          "//cabby/model/text:util",
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
          "//cabby/model/text:util",
          "//cabby/geo:regions",
          "//cabby/evals:utils"

    ],
)


py_test(
    name = "util_test",
    srcs = ["util_test.py"],
    python_version = "PY3",
    deps = [":util"],
)
