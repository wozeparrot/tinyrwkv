{
  description = "tinyrwkv: tiny implementation of RWKV in tinygrad";

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
                tokenizers
                torch
              ];
            python = pkgs.python311;
          in
            with pkgs; [
              (python.withPackages python-packages)
              llvmPackages_latest.clang
            ];
        };
      }
    );
}
