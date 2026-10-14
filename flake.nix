{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      rust-overlay,
    }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        overlays = [ rust-overlay.overlays.default ];
      };

      rust = pkgs.rust-bin.stable.latest.default.override {
        extensions = [ "rust-src" ];
      };

      rustPlatform = pkgs.makeRustPlatform {
        cargo = rust;
        rustc = rust;
      };

      # Use this flake's pins and toolchain with the shared package recipe.
      mkSwiv = targetCpu: pkgs.callPackage ./package.nix { inherit rustPlatform targetCpu; };
      swiv = mkSwiv null;

      libPath = pkgs.lib.makeLibraryPath swiv.libs;

      # for the uv scripts in scripts/
      pythonEnv = pkgs.python3.withPackages (
        ps: with ps; [
          uv
        ]
      );
    in
    {
      packages.${system} = {
        default = swiv;
        native = mkSwiv "native";
        x86-64-v3 = mkSwiv "x86-64-v3";
      };

      devShells.${system}.default = pkgs.mkShell {
        nativeBuildInputs = [
          rust
          pkgs.pkg-config
          pythonEnv
        ];

        buildInputs = swiv.libs ++ [
          # pkgs.vulkan-tools
          # pkgs.vulkan-tools-lunarg
          # pkgs.vulkan-extension-layer
          # pkgs.vulkan-validation-layers
        ];

        # wanted to avoid setting RUSTFLAGS for this but using LD_LIBRARY_PATH
        # makes checking whether builds work possibly confusing
        RUSTFLAGS = "-C link-arg=-Wl,-rpath,${libPath} -C target-cpu=native";

        env = {
          UV_PYTHON_DOWNLOADS = "never";
          UV_NO_MANAGED_PYTHON = "1";
          # manylinux wheels (numpy et al) need these at runtime; scoped to
          # libs the rust build doesn't link so rpath issues stay visible
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            pkgs.stdenv.cc.cc.lib
            pkgs.zlib
          ];
        };
      };
    };
}
