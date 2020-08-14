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
    name = 'extract_wikipedia_items',
    main = 'extract_wikipedia_items.py',
    srcs = ['extract_wikipedia_items.py'],
    deps = [':query'],
)