package(
    default_visibility = ["//visibility:public"],
)

py_library(
    name = 'extract',
    srcs = ['extract.py'],
)

py_library(
    name = 'wikigeo',
    srcs = ['wikigeo.py'],
)


py_test(
    name = "extract_test",
    srcs = ["extract_test.py"],
    python_version = "PY3",
    deps = [
        ":extract",
        ":wikigeo",
        "//cabby/data/wikidata:item",
        "//cabby/data/wikidata:query",
        "//cabby/data/wikipedia:item",
        "//cabby/data/wikipedia:query"


    ],
)


py_binary(
    name = 'extract_wikigeo_contexts',
    main = 'extract_wikigeo_contexts.py',
    srcs = ['extract_wikigeo_contexts.py'],
    deps = [
        ':extract',
        "//cabby/data/wikidata:item",
        "//cabby/data/wikidata:query",
        "//cabby/data/wikipedia:item",
        "//cabby/data/wikipedia:query",
        "//cabby/data:wikigeo"
    ],
)