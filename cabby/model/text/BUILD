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
        '//cabby/model/text/dual_encoder:train',
        '//cabby/model:datasets',
        '//cabby/model:dataset_item',
        '//cabby/model/text/dual_encoder:model',
        "//cabby/model:util",
        ':models'        

    ],
)


py_binary(
    name = 'models',
    main = 'models.py',
    srcs = ['models.py'],

)