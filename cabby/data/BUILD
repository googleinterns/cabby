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
    name = 'save',
    main = 'save.py',
    srcs = ['save.py'],
    deps = [':extract'],
)
