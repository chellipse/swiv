# swiv

A fast, minimal, GPU-accelerated image viewer written in Rust. Like
sxiv but wayland-native. Rendering via
[wgpu](https://github.com/gfx-rs/wgpu).

## Keybindings

There are two modes: **gallery** and **single-image**,
toggled via `RET`.

### Both modes

| Key | Action |
| --- | --- |
| `q` / `ESC` | quit |
| `RET` | toggle gallery / single-image mode |
| `h` / `left` | previous image |
| `l` / `right` | next image |
| `k` / `up` | move up one gallery row |
| `j` / `down` | move down one gallery row |
| `g` / `G` | first / last image |
| `f` | toggle fullscreen |

### Gallery

| Key | Action |
| --- | --- |
| `+` | fewer rows on screen (larger thumbnails) |
| `-` | more rows on screen (smaller thumbnails) |
| `=` | reset row count |
| scroll wheel | scroll rows |

### Single image

| Key | Action |
| --- | --- |
| `+` / `-` | zoom in / out |
| `=` | reset zoom |
| `r` | reset zoom and pan |
| scroll wheel | zoom (smooth on touchpads) |
| left-click drag | pan |

## Format support

| Format | Extensions |
| --- | --- |
| PNG | `png` |
| JPEG | `jpg`, `jpeg` |
| GIF | `gif` |
| WebP | `webp` |
| AVIF | `avif` |
| TIFF | `tiff`, `tif` |
| BMP | `bmp` |
| ICO | `ico` |
| Truevision TGA | `tga` |
| DirectDraw Surface | `dds` |
| Farbfeld | `farbfeld`, `ff` |
| Netpbm | `pnm`, `pbm`, `pgm`, `ppm`, `pam` |
| Radiance HDR | `hdr` |
| OpenEXR | `exr` |
| QOI | `qoi` |
| Photoshop | `psd`, `psb` |

### Camera raws

| Vendor | Extensions |
| --- | --- |
| Adobe DNG | `dng` |
| Apple QuickTake | `qtk` |
| ARRI | `ari` |
| Canon | `crw`, `cr2`, `cr3`, `crm` |
| Epson | `erf` |
| Fujifilm | `raf` |
| Hasselblad | `3fr`, `fff` |
| Kodak | `dcr`, `dcs`, `kdc` |
| Leaf | `mos` |
| Leica | `rwl` |
| Mamiya | `mef` |
| Minolta | `mrw` |
| Nikon | `nef`, `nrw` |
| Olympus | `orf`, `ori` |
| Panasonic / Leica | `rw2`, `raw` |
| Pentax | `pef` |
| Phase One | `iiq` |
| Samsung | `srw` |
| Sigma | `x3f` |
| Sony | `arw` |

By default single-image uses the raw's embedded preview jpeg. Passing
`--raw-mode` will cause single-image mode to do a proper demosaic of
the raw sensor data. Note: swiv does not implement the default preprocessing
settings that programs like
[Darktable](https://github.com/darktable-org/darktable) do, so the
raw won't look very good. Gallery thumbnails always use the embedded
previews.

## License

[MIT](LICENSE)
