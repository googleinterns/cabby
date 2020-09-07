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
        "//cabby/geo:walk",
        "//cabby/geo/map_processing:map_structure",

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
    deps = [
        ':walk',
        ':util',
        "//cabby:logger"
        ],
    
)

py_binary(
    name = 'sample_poi',
    main = 'sample_poi.py',
    srcs = ['sample_poi.py'],
    deps = [
        ':walk',
        ':util',
        "//cabby:logger"
        ],
)