package(
    default_visibility = ["//visibility:public"],
)

py_binary(
    name = 'create_graph_embedding',
    srcs = ['create_graph_embedding.py'],
    deps = [
        "//cabby/geo:regions",
        ":utils",
    ],
)

py_library(
    name = 'utils',
    srcs = ['utils.py'],
    deps = [
    "//cabby/data/wikidata:query",
    "//cabby/geo:regions",
    "//cabby/geo:util",
    "//cabby/geo/map_processing:map_structure",

    ],
)


py_test(
    name = "graph_embedding_test",
    srcs = ["graph_embedding_test.py"],
    python_version = "PY3",
    deps = [
        "//cabby/geo:regions",
        ":utils",
        '//cabby/geo/map_processing:map_structure'
    ],
)