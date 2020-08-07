package(
    default_visibility = ["//visibility:public"],
)

py_library(
    name = 'speak',
    srcs = ['speak.py'],
)

py_test(
    name = "speak_test",
    srcs = ["speak_test.py"],
    python_version = "PY3",
    deps = [
        ":speak"
    ],
)

py_binary(
    name = 'generate_rvs',
    main = 'generate_rvs.py',
    srcs = ['generate_rvs.py'],
    deps = [':speak'],
)
