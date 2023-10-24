{
  description = "";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";

    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    tinygrad = {
      url = "github:geohot/tinygrad";
      flake = false;
    };
  };

  outputs = inputs @ {
    nixpkgs,
    flake-utils,
    fenix,
    ...
  }:
    flake-utils.lib.eachDefaultSystem
    (
      system: let
        pkgs = nixpkgs.legacyPackages.${system};
        rust-toolchain = with fenix.packages.${system};
          combine [
            minimal.cargo
            minimal.rustc
            targets."aarch64-unknown-linux-musl".latest.rust-std
            targets."x86_64-unknown-linux-musl".latest.rust-std
          ];
      in {
        devShell = let
          wgpu-native = pkgs.rustPlatform.buildRustPackage rec {
            pname = "wgpu-native";
            version = "0.14.2.3";
            src = pkgs.fetchFromGitHub {
              owner = "gfx-rs";
              repo = pname;
              rev = "v${version}";
              fetchSubmodules = true;
              sha256 = "sha256-MkWU5STYKmyA2EamwXtGzmz4dcL1aty1NkG8yOylUR8=";
            };
            cargoLock = {
              lockFile = ./nix/wgpu-native_Cargo.lock;
              outputHashes = {
                "naga-0.10.0" = "sha256-CqsIPHHdrzollxhc6/eFSf91nBiHzMuGm4CxsKagKMo=";
                "wgpu-core-0.14.2" = "sha256-UNJ9rD6vFOja6g9/7hxR5u3cl8rwpjlT8UPrZI4azgE=";
              };
            };
            nativeBuildInputs = with pkgs; [
              rustPlatform.bindgenHook
            ];
          };
        in
          pkgs.mkShell {
            packages = let
              python-packages = p: let
                wgpu-py = p.buildPythonPackage rec {
                  pname = "wgpu-py";
                  version = "0.9.4";
                  src = pkgs.fetchFromGitHub {
                    owner = "pygfx";
                    repo = pname;
                    rev = "v${version}";
                    sha256 = "sha256-1h+l4Fa3ooM1ifUjSyte6qCa1LrS2XBlkcHTBDCR9Kw=";
                  };
                  doCheck = false;
                  propagatedBuildInputs = with p; [
                    cffi
                  ];
                };
                tinygrad = p.buildPythonPackage {
                  pname = "tinygrad";
                  version = inputs.tinygrad.shortRev;
                  src = inputs.tinygrad;
                  doCheck = false;
                  propagatedBuildInputs = with p; [
                    networkx
                    numpy
                    pillow
                    pyopencl
                    pyyaml
                    requests
                    tqdm
                  ];
                };
              in
                with p; [
                  llvmlite
                  mido
                  mypy
                  numpy
                  pydot
                  tinygrad
                  tokenizers
                  torch
                  tqdm
                  wandb
                  wgpu-py
                ];
              llvmPackages = pkgs.llvmPackages_15;
              python = pkgs.python311;
            in
              with pkgs; [
                (python.withPackages python-packages)
                rust-toolchain
                llvmPackages.clang
                llvmPackages.libunwind

                # needed for GRAPH=1 to work
                graphviz
              ];

            shellHook = ''
              export WGPU_LIB_PATH=${wgpu-native}/lib/libwgpu_native.so
              export LD_LIBRARY_PATH=${pkgs.vulkan-loader}/lib
            '';
          };
      }
    );
}
