package(
    default_visibility = ["//visibility:public"],
)

py_library(
    name = 'graph',
    srcs = ['graph.py'],
)

py_library(
    name = 'map_structure',
    srcs = ['map_structure.py'],
)


py_test(
    name = 'map_test',
    srcs = ['map_test.py'],
    python_version = 'PY3',
    deps = [
        ':map_structure'
    ],
)

py_binary(
    name = 'map_processor',
    main = 'map_processor.py',
    srcs = ['map_processor.py'],
    deps = [
        ':map_structure',
        ':graph',
        ],
)
