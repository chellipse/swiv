let
  # TODO give this derivation proper shell/build attributes
  rust_overlay = import (
    builtins.fetchTarball {
      url = "https://github.com/oxalica/rust-overlay/archive/0adf92c70d23fb4f703aea5d3ebb51ac65994f7f.tar.gz"; # master
      sha256 = "1k6rj55mradsimk5c8z97qxz0rmhsbz3p1d2n4lqk1qqg9b7g6n4"; # 2025-08-28T13·14+00
    }
  );

  pkgs = import (fetchTarball {
    url = "https://github.com/NixOS/nixpkgs/archive/8a6d5427d99ec71c64f0b93d45778c889005d9c2.tar.gz"; # nixos-unstable
    sha256 = "1hqav9bvqiij8vzzg7j9c77m3z2hkwh3x5x4hvkzp9d6fkrgigkj"; # 2025-08-28T13·14+00
  }) { overlays = [ rust_overlay ]; };

  rust = pkgs.rust-bin.stable.latest.default.override {
    extensions = [ "rust-src" ];
  };

  # TODO test what deps we need to make this build on linux/X and possibly windows
  libPath =
    with pkgs;
    lib.makeLibraryPath [
      wayland
      libxkbcommon
      # wayland.dev
      # xorg.libX11
      # xorg.libXcursor
      # xorg.libXrandr
      # xorg.libXi
      # alsa-lib
      # fontconfig
      # freetype
      # shaderc
      # directx-shader-compiler
      # libGL
      vulkan-headers
      vulkan-loader
      vulkan-tools
      vulkan-tools-lunarg
      vulkan-extension-layer
      vulkan-validation-layers
    ];
in
pkgs.mkShell {
  nativeBuildInputs = [
    rust
  ];

  LD_LIBRARY_PATH = libPath;
}
