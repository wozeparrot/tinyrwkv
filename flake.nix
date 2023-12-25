{
  description = "";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";

    tinygrad.url = "github:wozeparrot/tinygrad-nix";
  };

  outputs = inputs @ {
    nixpkgs,
    flake-utils,
    ...
  }:
    flake-utils.lib.eachDefaultSystem
    (
      system: let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [
            inputs.tinygrad.overlays.default
          ];
        };
      in {
        devShell = pkgs.mkShell {
          packages = let
            python-packages = p:
              with p; [
                tinygrad
                llvmlite
                mido
                mypy
                numpy
                pydot
                tokenizers
                torch
                tqdm
                wandb
              ];
            llvmPackages = pkgs.llvmPackages_15;
            python = pkgs.python311;
          in
            with pkgs; [
              (python.withPackages python-packages)
              cargo
              llvmPackages.clang
              llvmPackages.libunwind
              clinfo

              # needed for GRAPH=1 to work
              graphviz
            ];
        };
      }
    );
}
