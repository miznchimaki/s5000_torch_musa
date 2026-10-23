def define_targets(rules, extra_defines=[]):
    rules.cc_library(
        name = "musa",
        srcs = rules.glob(
            [
                "*.cpp",
                "impl/*.cpp",
            ],
            exclude = [
                "test/**/*.cpp",
            ],
        ),
        hdrs = rules.glob(
            [
                "*.h",
                "impl/*.h",
            ],
            exclude = [
                "MUSAMacros.h",
            ],
        ),
        defines = ["USE_CUDA"] + extra_defines,
        linkstatic = True,
        local_defines = ["C10_BUILD_MAIN_LIB"],
        target_compatible_with = rules.requires_cuda_enabled(),
        visibility = ["//visibility:public"],
        deps = [
            ":Macros",
            "//c10/core:base",
            "//c10/macros",
            "//c10/util:base",
            "@musa",
        ],
        # This library uses registration. Don't let registered
        # entities be removed.
        alwayslink = True,
    )

    rules.cc_library(
        name = "Macros",
        srcs = [":musa_cmake_macros"],
        hdrs = ["MUSAMacros.h"],
        linkstatic = True,
        local_defines = ["C10_BUILD_MAIN_LIB"],
        visibility = ["//visibility:public"],
    )

    rules.cmake_configure_file(
        name = "musa_cmake_macros",
        src = "impl/musa_cmake_macros.h.in",
        out = "impl/musa_cmake_macros.h",
        definitions = [],
    )

    rules.filegroup(
        name = "headers",
        srcs = rules.glob(
            [
                "*.h",
                "impl/*.h",
            ],
            exclude = [
            ],
        ),
        visibility = ["//c10:__pkg__"],
    )
