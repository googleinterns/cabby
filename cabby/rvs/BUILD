package(
    default_visibility = ["//visibility:public"],
)

py_library(
    name = 'observe',
    srcs = ['observe.py'],
)

py_library(
    name = 'speak',
    srcs = ['speak.py'],
    deps = [
        '//cabby/geo:directions',
    ]
)

py_test(
    name = "speak_test",
    srcs = ["speak_test.py"],
    python_version = "PY3",
    deps = [
        ":speak",
        '//cabby/geo:directions',
    ],
)

py_binary(
    name = 'generate_rvs',
    main = 'generate_rvs.py',
    srcs = ['generate_rvs.py'],
    deps = [
        ':observe',
        ':speak',
        "//cabby/data/wikidata:item",
        "//cabby/data/wikidata:query",
        "//cabby/geo:directions",
    ],
)
