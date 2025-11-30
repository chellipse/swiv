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

      inputs = with pkgs; [
        wayland
        libxkbcommon
        vulkan-headers
        vulkan-loader
        vulkan-tools
        vulkan-tools-lunarg
        vulkan-extension-layer
        vulkan-validation-layers
        dav1d
      ];

      libPath = pkgs.lib.makeLibraryPath inputs;

    in
    {
      packages.${system}.default = rustPlatform.buildRustPackage rec {
        pname = "untitled-image-viewer";
        version = "0.1.0";

        src = pkgs.lib.cleanSourceWith {
          src = ./.;
          filter =
            path: type:
            (pkgs.lib.hasSuffix ".wgsl" path)
            || (pkgs.lib.cleanSourceFilter path type)
            || (builtins.match ".*/(Cargo\\.toml|Cargo\\.lock|src/.*)" path != null);
        };

        cargoLock.lockFile = ./Cargo.lock;

        buildInputs = inputs;

        nativeBuildInputs = [ pkgs.pkg-config ];

        postFixup = ''
          patchelf --set-rpath ${libPath} $out/bin/${pname}
        '';
      };

      devShells.${system}.default = pkgs.mkShell {
        nativeBuildInputs = [
          rust
          pkgs.pkg-config
        ];

        buildInputs = inputs;

        # wanted to avoid setting RUSTFLAGS for this but using LD_LIBRARY_PATH
        # makes checking whether builds work possibly confusing
        RUSTFLAGS = "-C link-arg=-Wl,-rpath,${libPath}";
      };
    };
}
