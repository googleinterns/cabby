package(
    default_visibility = ["//visibility:public"],
)

py_library(
    name = 'observe',
    srcs = ['observe.py'],
)

py_test(
    name = "observe_test",
    srcs = ["observe_test.py"],
    python_version = "PY3",
    deps = [
        ":observe",
        "//cabby/data/wikidata:item",
        "//cabby/data/wikidata:query",
        '//cabby/geo:regions',
    ],
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

py_test(
    name = "templates_test",
    srcs = ["templates_test.py"],
    python_version = "PY3",
    deps = [
        ":templates",
        '//cabby/geo:geo_item',
        '//cabby/geo:util',

    ],
)

py_binary(
    name = 'generate_rvs',
    main = 'generate_rvs.py',
    srcs = ['generate_rvs.py'],
    deps = [
        ":templates",
        '//cabby/geo:geo_item',
        "//cabby/geo:walk",


    ],
)

py_library(
    name = 'templates',
    srcs = ['templates.py'],
        deps = [
        '//cabby/geo:geo_item',
        '//cabby/geo:walk',
         '//cabby/geo:util',

    ],
)
