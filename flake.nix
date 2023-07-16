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
        devShell = pkgs.mkShell {
          packages = let
            python-packages = p: let
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
                mypy
                numpy
                pydot
                tinygrad
                tokenizers
                torch
                tqdm
                wandb
                mido
              ];
            llvmPackages = pkgs.llvmPackages_15;
            python = pkgs.python310;
          in
            with pkgs; [
              (python.withPackages python-packages)
              rust-toolchain
              llvmPackages.clang
              llvmPackages.libunwind

              # needed for GRAPH=1 to work
              graphviz
            ];
        };
      }
    );
}
