# Cabby: Geographic vision-and-language navigation

licenses(["notice"])

exports_files(["LICENSE"])

package(
    default_visibility = ["//visibility:public"],
)


py_library(
    name = 'dataset',
    srcs = ['dataset.py'],
    deps = [
          "//cabby/model:datasets",
    ],
)


py_library(
    name = 'run',
    srcs = ['run.py'],
)


py_binary(
    name = 'entity_recognition',
    main = 'entity_recognition.py',
    srcs = ['entity_recognition.py'],
    deps = [
          "//cabby/geo:regions",
          "//cabby/model/landmark_recognition:dataset",
          "//cabby/model/landmark_recognition:run",

    ],
)