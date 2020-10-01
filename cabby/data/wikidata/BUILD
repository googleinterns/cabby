package(
    default_visibility = ["//visibility:public"],
)

py_library(
    name = 'item',
    srcs = ['item.py'],
)

py_library(
    name = 'info_item',
    srcs = ['info_item.py'],
)

py_library(
    name = 'query',
    srcs = ['query.py'],
    deps = [
        "//cabby/geo:util",
        "//cabby/geo/map_processing:map_structure"

    ]
)

py_test(
    name = "query_test",
    srcs = ["query_test.py"],
    python_version = "PY3",
    deps = [
        ":query",
        "//cabby/geo:util",
        "//cabby/geo/map_processing:map_structure"
    ],
)

py_binary(
    name = 'extract_geofenced_wikidata_items',
    main = 'extract_geofenced_wikidata_items.py',
    srcs = ['extract_geofenced_wikidata_items.py'],
    deps = [
        ":query",
        "//cabby/geo:util",
        "//cabby/geo/map_processing:map_structure"
    ],
)