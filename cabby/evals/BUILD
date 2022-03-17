package(
    default_visibility = ["//visibility:public"],
)
py_library(
    name = 'utils',
    srcs = ['utils.py'],
    deps = [
        "//cabby:logger",
    ]
)
py_test(
    name = 'utils_test',
    srcs = ['utils_test.py'],
    python_version = 'PY3',
    data = glob(["testdata/**"]),
    deps = [
        ":utils",
    ],
)        
py_binary(
    name = 'eval_geo_model',
    srcs = ['eval_geo_model.py'],
    deps = [
        ":utils",
        "//cabby:logger",
        "//cabby/model:util",
    ],
)