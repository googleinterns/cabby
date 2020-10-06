package(
    default_visibility = ["//visibility:public"],
)

py_library(
    name = 'extract',
    srcs = ['extract.py'],
    deps = [
        ":wikigeo",
        "//cabby/data/wikidata:item",
        "//cabby/data/wikidata:query",
        "//cabby/data/wikipedia:item",
        "//cabby/data/wikipedia:query",
        "//cabby/data/wikidata:info_item",
        '//cabby/geo/map_processing:map_structure'


    ],

)

py_library(
    name = 'wikigeo',
    srcs = ['wikigeo.py'],
)

py_library(
    name = 'osm_item',
    srcs = ['osm_item.py'],
)


py_test(
    name = "extract_test",
    srcs = ["extract_test.py"],
    python_version = "PY3",
    deps = [
        ":extract",
        ":wikigeo",
        ":osm_item",
        "//cabby/data/wikidata:item",
        "//cabby/data/wikidata:query",
        "//cabby/data/wikipedia:item",
        "//cabby/data/wikipedia:query",
        "//cabby/data/wikidata:info_item",
        '//cabby/geo/map_processing:map_structure'


    ],
)


py_binary(
    name = 'extract_wikigeo_contexts',
    main = 'extract_wikigeo_contexts.py',
    srcs = ['extract_wikigeo_contexts.py'],
    deps = [
        ':extract',
        ":osm_item",
        "//cabby/data/wikidata:item",
        "//cabby/data/wikidata:query",
        "//cabby/data/wikipedia:item",
        "//cabby/data/wikipedia:query",
        "//cabby/data:wikigeo"
    ],

)



py_binary(
    name = 'extract_wikigeo_contexts_expansive',
    main = 'extract_wikigeo_contexts_expansive.py',
    srcs = ['extract_wikigeo_contexts_expansive.py'],
    deps = [
        ':extract',
         ":osm_item",
        "//cabby/data/wikidata:item",
        "//cabby/data/wikidata:info_item",
        "//cabby/data/wikidata:query",
        "//cabby/data/wikipedia:item",
        "//cabby/data/wikipedia:query",
        "//cabby/data:wikigeo",
        '//cabby/geo:regions',
        '//cabby/geo/map_processing:map_structure'

    ],

)