// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "microgpt.swift",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "microgpt.swift", targets: ["microgpt.swift"]),
        .executable(name: "translategemma-swift", targets: ["translategemma-swift"]),
        .executable(name: "gguf-inspect", targets: ["gguf-inspect"]),
        .executable(name: "translategemma-native-tokenize", targets: ["translategemma-native-tokenize"]),
        .library(name: "GGUFCore", targets: ["GGUFCore"]),
    ],
    dependencies: [
        .package(path: "Vendor/LocalLLMClient"),
    ],
    targets: [
        .executableTarget(
            name: "microgpt.swift"
        ),
        .executableTarget(
            name: "translategemma-swift",
            dependencies: [
                .product(name: "LocalLLMClient", package: "LocalLLMClient"),
                .product(name: "LocalLLMClientLlama", package: "LocalLLMClient"),
            ],
            swiftSettings: [
                .interoperabilityMode(.Cxx)
            ]
        ),
        .target(
            name: "GGUFCore"
        ),
        .executableTarget(
            name: "gguf-inspect",
            dependencies: ["GGUFCore"]
        ),
        .executableTarget(
            name: "translategemma-native-tokenize",
            dependencies: ["GGUFCore"]
        ),
    ]
)
