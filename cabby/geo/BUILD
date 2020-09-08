package(
    default_visibility = ["//visibility:public"],
)

py_library(
    name = 'walk',
    srcs = ['walk.py'],
    deps = [
    "//cabby/geo/map_processing:map_structure",

    ],
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
        ":walk",
        "//cabby/geo/map_processing:map_structure",
    ],
)

py_library(
    name = 'directions',
    srcs = ['directions.py'],
    deps = [
        ':walk',
        ':util',
        "//cabby:logger",
        "//cabby/geo/map_processing:map_structure",

        ],
)




py_binary(
    name = 'geo_computation',
    main = 'geo_computation.py',
    srcs = ['geo_computation.py'],
    deps = [
        ':walk',
        ':util',
        "//cabby:logger",

    ],
)
