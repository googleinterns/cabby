package(
    default_visibility = ["//visibility:public"],
)

py_library(
    name = 'wikidata',
    srcs = ['wikidata.py'],
)

py_test(
    name = "wikidata_test",
    srcs = ["wikidata_test.py"],
    python_version = "PY3",
    deps = [
        ":wikidata"
    ],
)

py_binary(
    name = 'extract_wikidata',
    main = 'extract_wikidata.py',
    srcs = ['extract_wikidata.py'],
    deps = [':wikidata'],
)
