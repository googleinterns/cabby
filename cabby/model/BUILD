# Cabby: Geographic vision-and-language navigation

licenses(["notice"])

exports_files(["LICENSE"])

package(
    default_visibility = ["//visibility:public"],
)

py_library(
    name = 'util',
    srcs = ['util.py'],    
)

py_test(
    name = "util_test",
    srcs = ["util_test.py"],
    python_version = "PY3",
    deps = [":util"],
)
