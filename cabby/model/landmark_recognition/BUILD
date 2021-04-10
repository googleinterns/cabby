# Cabby: Geographic vision-and-language navigation

licenses(["notice"])

exports_files(["LICENSE"])

package(
    default_visibility = ["//visibility:public"],
)


py_library(
    name = 'dataset_bert',
    srcs = ['dataset_bert.py'],
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
          "//cabby/model/landmark_recognition:dataset_bert",
          "//cabby/model/landmark_recognition:run",

    ],
)


py_binary(
    name = 'pivot_recognition',
    main = 'pivot_recognition.py',
    srcs = ['pivot_recognition.py'],
    deps = [
          "//cabby/geo:regions",
          "//cabby/model/landmark_recognition:dataset_bert",
          "//cabby/model/landmark_recognition:run",

    ],
)