package(
    default_visibility = ["//visibility:public"],
)

py_library(
    name = 'walk',
    srcs = ['walk.py'],
)

py_library(
    name = 'util',
    srcs = ['util.py'],
)

py_test(
    name = "walk_test",
    srcs = ["walk_test.py"],
    python_version = "PY3",
    deps = [
        ":walk"
    ],
)

py_library(
    name = 'directions',
    srcs = ['directions.py'],
    deps = [':util'],
)


py_binary(
    name = 'geo_computation',
    main = 'geo_computation.py',
    srcs = ['geo_computation.py'],
    deps = [':walk'],
)
