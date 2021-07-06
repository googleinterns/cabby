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
    name = 'dataset_t5',
    srcs = ['dataset_t5.py'],
    deps = [
          "//cabby/model:datasets",
    ],
)


py_library(
    name = 'run',
    srcs = ['run.py'],
)


py_library(
    name = 'run_t5',
    srcs = ['run_t5.py'],
)


py_binary(
    name = 'pivot_recognition',
    main = 'pivot_recognition.py',
    srcs = ['pivot_recognition.py'],
    deps = [
          "//cabby/geo:regions",
          "//cabby/model/landmark_recognition:dataset_bert",
          "//cabby/model/landmark_recognition:run",
          "//cabby/geo:walk"

    ],
)

py_binary(
    name = 'pivot_recognition_t5',
    main = 'pivot_recognition_t5.py',
    srcs = ['pivot_recognition_t5.py'],
    deps = [
          "//cabby/geo:regions",
          "//cabby/model/landmark_recognition:dataset_t5",
          "//cabby/model/landmark_recognition:run_t5",
          "//cabby/geo:walk"

    ],
)

