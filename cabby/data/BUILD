package(
    default_visibility = ["//visibility:public"],
)

py_library(
    name = 'extract',
    srcs = ['extract.py'],
)

py_test(
    name = "extract_test",
    srcs = ["extract_test.py"],
    python_version = "PY3",
    deps = [
        ":extract"
    ],
)

py_binary(
    name = 'extract_wikigeo_contexts',
    main = 'extract_wikigeo_contexts.py',
    srcs = ['extract_wikigeo_contexts.py'],
    deps = [':extract'],
)
