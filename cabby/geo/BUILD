package(
    default_visibility = ["//visibility:public"],
)

py_library(
    name = 'walk',
    srcs = ['walk.py'],
    deps = [
        ':util',
        ':geo_item',
        ':osm',
        '//cabby/rvs:item',
        '//cabby/geo/map_processing:map_structure',
        '//cabby/geo:regions'

    ],

)

py_library(
    name = 'osm',
    srcs = ['osm.py'],
)

py_library(
    name = 'geo_item',
    srcs = ['geo_item.py'],
)


py_library(
    name = 'util',
    srcs = ['util.py'],
)

py_library(
    name = 'regions',
    srcs = ['regions.py'],
)

py_test(
    name = "walk_test",
    srcs = ["walk_test.py"],
    python_version = "PY3",
    deps = [
        '//cabby/rvs:item',
        ":walk",
        ':geo_item',
        "//cabby/geo/map_processing:map_structure",
        '//cabby/geo:regions'
    ],
		
)

py_library(
    name = 'directions',
    srcs = ['directions.py'],
    deps = [
        ':walk',
        ':util',
        ':geo_item',
        "//cabby:logger",
        "//cabby/geo/map_processing:map_structure",
        ],
)

py_binary(
    name = 'sample_poi',
    main = 'sample_poi.py',
    srcs = ['sample_poi.py'],
    deps = [
        ':walk',
        ':util',
        '//cabby/rvs:item',
        "//cabby:logger",
        '//cabby/geo:regions'
        ],
)