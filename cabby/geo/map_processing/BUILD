package(
    default_visibility = ["//visibility:public"],
)

py_library(
    name = 'graph',
    srcs = ['graph.py'],
)

py_library(
    name = 'edge',
    srcs = ['edge.py'],
)

py_library(
    name = 'map_structure',
    srcs = ['map_structure.py'],
    deps = [
        ':graph',
        ':edge',
        '//cabby/geo:util',
        '//cabby:logger',

    ]
)


py_test(
    name = 'map_test',
    srcs = ['map_test.py'],
    python_version = 'PY3',
    deps = [
        ':edge',
        "//cabby/geo/map_processing:map_structure",
        "//cabby/geo/map_processing:graph",
        "//cabby/geo:util",
        "//cabby:logger",

    ],
)        


py_binary(
    name = 'map_processor',
    main = 'map_processor.py',
    srcs = ['map_processor.py'],
    deps = [
        ':map_structure',
        ':graph',
        ':edge',
        '//cabby:logger',
        ],
)