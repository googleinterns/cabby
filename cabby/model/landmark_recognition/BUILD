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
    name = 'geo_pivot_recognition',
    main = 'geo_pivot_recognition.py',
    srcs = ['geo_pivot_recognition.py'],
    deps = [
          "//cabby/geo:regions",
          "//cabby/model/landmark_recognition:dataset_bert",
          "//cabby/model/landmark_recognition:run",
          "//cabby/geo:walk",
          "//cabby/model/text/dual_encoder:run",
          "//cabby/model/text/dual_encoder:dataset_rvs",
          "//cabby/model/text/dual_encoder:dataset_item",
          "//cabby/model/text/dual_encoder:model",
          "//cabby/model/text:util",
          "//cabby/evals:utils"

    ],
)

