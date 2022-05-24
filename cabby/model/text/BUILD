# Cabby: Geographic vision-and-language navigation

licenses(["notice"])

exports_files(["LICENSE"])

package(
    default_visibility = ["//visibility:public"],
)


py_binary(
    name = 'model_trainer',
    main = 'model_trainer.py',
    srcs = ['model_trainer.py'],
    deps = [
        '//cabby/model/text:train',
        '//cabby/model:datasets',
        '//cabby/model:dataset_item',
        "//cabby/model:util",
        "//cabby/geo:util",
        ':models'        

    ],
)

py_binary(
    name = 'model_trainer_multitask',
    main = 'model_trainer_multitask.py',
    srcs = ['model_trainer_multitask.py'],
    deps = [
        '//cabby/model/text:train',
        '//cabby/model:datasets',
        '//cabby/model:dataset_item',
        "//cabby/model:util",
        "//cabby/geo:util",
        ':models'        

    ],
)



py_binary(
    name = 'models',
    main = 'models.py',
    srcs = ['models.py'],
    deps = [
        "//cabby/model:util",

],

)

py_library(
    name = 'train',
    srcs = ['train.py'],    
    deps = [
        "//cabby/model:util",
        "//cabby/evals:utils",

],
)