package(
    default_visibility = ["//visibility:public"],
)

py_library(
    name = 'query',
    srcs = ['query.py'],
)

py_test(
    name = "query_test",
    srcs = ["query_test.py"],
    python_version = "PY3",
    deps = [
        ":query"
    ],
)

py_binary(
    name = 'extract_wikidata',
    main = 'extract_wikidata.py',
    srcs = ['extract_wikidata.py'],
    deps = [':query'],
)