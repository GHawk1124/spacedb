{
  description = "spacedb: Rust app built with crane + Python dev shell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    crane.url = "github:ipetkov/crane";
    rust-overlay.url = "github:oxalica/rust-overlay";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
    crane,
    rust-overlay,
  }:
    flake-utils.lib.eachDefaultSystem (system: let
      overlays = [(import rust-overlay)];
      pkgs = import nixpkgs {
        inherit system overlays;
      };

      rustToolchain = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;
      craneLib = (crane.mkLib pkgs).overrideToolchain rustToolchain;

      src = craneLib.cleanCargoSource ./.;

      runtimeLibs = with pkgs; [
        libGL
        libxkbcommon
        openssl
        vulkan-loader
        wayland
        xorg.libX11
        xorg.libXcursor
        xorg.libXi
        xorg.libXinerama
        xorg.libXrandr
        xorg.libxcb
      ];

      commonArgs = {
        inherit src;
        strictDeps = true;
        nativeBuildInputs = with pkgs; [
          git
          pkg-config
        ];
        buildInputs = runtimeLibs;
        OPENSSL_NO_VENDOR = 1;
        OPENSSL_DIR = "${pkgs.openssl.dev}";
        OPENSSL_LIB_DIR = "${pkgs.openssl.out}/lib";
        OPENSSL_INCLUDE_DIR = "${pkgs.openssl.dev}/include";
      };

      cargoArtifacts = craneLib.buildDepsOnly commonArgs;

      spacedbUnwrapped = craneLib.buildPackage (
        commonArgs
        // {
          inherit cargoArtifacts;
          doCheck = false;
        }
      );

      spacedb = pkgs.symlinkJoin {
        name = "spacedb";
        paths = [spacedbUnwrapped];
        nativeBuildInputs = [pkgs.makeWrapper];
        postBuild = ''
          wrapProgram $out/bin/spacedb \
            --prefix LD_LIBRARY_PATH : "${pkgs.lib.makeLibraryPath runtimeLibs}"
        '';
      };

      pythonEnv = pkgs.python313;
    in {
      packages = {
        default = spacedb;
        spacedb = spacedb;
      };

      apps = {
        default = flake-utils.lib.mkApp {
          drv = spacedb;
        };
      };

      devShells.default = pkgs.mkShell {
        inputsFrom = [spacedb];
        OPENSSL_NO_VENDOR = 1;
        OPENSSL_DIR = "${pkgs.openssl.dev}";
        OPENSSL_LIB_DIR = "${pkgs.openssl.out}/lib";
        OPENSSL_INCLUDE_DIR = "${pkgs.openssl.dev}/include";
        packages = with pkgs; [
          rustToolchain
          cargo-watch
          pythonEnv
          uv
        ];

        shellHook = ''
          echo "Rust: cargo build / cargo run"
          echo "Python script: uv run python space_objects.py"
          echo "(first time) run: uv sync"
        '';
      };
    });
}
