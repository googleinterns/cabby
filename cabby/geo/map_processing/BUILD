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
         '//cabby/geo:regions'

    ]
)


py_test(
    name = 'map_test',
    srcs = ['map_test.py'],
    python_version = 'PY3',
    deps = [
        ':edge',
        ":map_structure",
        ":graph",
        "//cabby/geo:util",
        "//cabby:logger",
        '//cabby/geo:regions'


    ],
)        


py_binary(
    name = 'get_geofenced_pois',
    main = 'get_geofenced_pois.py',
    srcs = ['get_geofenced_pois.py'],
    deps = [
        ':map_structure',
        ':graph',
        ':edge',
        '//cabby:logger',
        '//cabby/geo:regions'
        ],
)