# Consumers use callPackage with their own nixpkgs to avoid loading the host's
# /run/opengl-driver Vulkan driver against an older glibc than it was built with.
{
  lib,
  rustPlatform,
  pkg-config,
  wayland,
  libxkbcommon,
  vulkan-headers,
  vulkan-loader,
  dav1d,
  # rustc -C target-cpu value, e.g. "native" or "x86-64-v3".
  targetCpu ? null,
}:

let
  libs = [
    wayland
    libxkbcommon
    vulkan-headers
    vulkan-loader
    dav1d
  ];
in
rustPlatform.buildRustPackage {
  pname = "swiv";
  version = "0.1.0";

  src = lib.fileset.toSource {
    root = ./.;
    fileset = lib.fileset.unions [
      ./Cargo.toml
      ./Cargo.lock
      ./src
    ];
  };

  cargoLock.lockFile = ./Cargo.lock;

  nativeBuildInputs = [ pkg-config ];

  buildInputs = libs;

  RUSTFLAGS = lib.optionalString (targetCpu != null) "-C target-cpu=${targetCpu}";

  # wgpu and winit dlopen the Vulkan loader, Wayland, and xkbcommon.
  postFixup = ''
    patchelf --add-rpath ${lib.makeLibraryPath libs} $out/bin/swiv
  '';

  passthru = { inherit libs; };
}
