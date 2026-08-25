//! Camera RAW backend: viewer-grade support that serves the embedded
//! camera-rendered preview JPEGs via rawler instead of developing sensor
//! data. Previews decode in milliseconds (a develop takes seconds) and
//! show the camera's own rendering — color science, tone curve, lens
//! corrections — which is what a photographer expects a viewer to show.

use std::{
    collections::HashSet,
    fs::File,
    io::{BufReader, Read, Seek, SeekFrom},
    path::Path,
};

use anyhow::{Result, anyhow};
use image::{DynamicImage, metadata::Orientation};
use rawler::{
    decoders::{Decoder, RawDecodeParams, supported_extensions},
    get_decoder,
    imgop::develop::RawDevelop,
    rawsource::RawSource,
};

/// How Full-resolution raws are rendered
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum RawMode {
    /// Demosaic + develop the actual sensor data (slow, neutral rendering)
    Develop,
    /// The camera's embedded preview jpeg (fast, the camera's own rendering)
    Preview,
}

pub fn is_raw(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|ext| {
            supported_extensions()
                .iter()
                .any(|supported| supported.eq_ignore_ascii_case(ext))
        })
}

pub struct Preview {
    pub image: DynamicImage,
    /// Set when the preview is not sRGB (cameras only embed sRGB or
    /// Adobe RGB)
    pub icc: Option<Vec<u8>>,
    /// Previews are stored unrotated; the caller must apply this
    pub orientation: Option<Orientation>,
}

/// Decodes the smallest embedded preview whose long edge covers
/// `min_edge`, or the largest available one for None
pub fn decode_preview(path: &Path, min_edge: Option<u32>) -> Result<Preview> {
    let source = RawSource::new(path)?;
    let decoder = get_decoder(&source)?;
    let params = RawDecodeParams::default();

    let covers = |img: &DynamicImage| match min_edge {
        Some(edge) => img.width().max(img.height()) >= edge,
        None => true,
    };

    // embedded images by convention: thumbnail < preview < full
    let image = match min_edge {
        Some(_) => decoder
            .thumbnail_image(&source, &params)
            .ok()
            .flatten()
            .filter(&covers)
            .or_else(|| {
                decoder
                    .preview_image(&source, &params)
                    .ok()
                    .flatten()
                    .filter(&covers)
            })
            .or_else(|| {
                largest_embedded(decoder.as_ref(), &source, &params, path)
            }),
        None => largest_embedded(decoder.as_ref(), &source, &params, path),
    }
    // nothing covering the wanted size: any preview at all beats nothing,
    // the decoder never upscales
    .or_else(|| decoder.preview_image(&source, &params).ok().flatten())
    .or_else(|| decoder.thumbnail_image(&source, &params).ok().flatten())
    .ok_or_else(|| {
        anyhow!("no embedded preview found (raw develop is not supported)")
    })?;

    let (icc, orientation) = exif_extras(decoder.as_ref(), &source, &params);

    Ok(Preview { image, icc, orientation })
}

/// Demosaics and develops the actual sensor data to an sRGB image.
/// Seconds of CPU and >0.5GB of f32 buffers for modern sensors — callers
/// should hold the huge-decode lock.
pub fn develop(path: &Path) -> Result<Preview> {
    let source = RawSource::new(path)?;
    let decoder = get_decoder(&source)?;
    let params = RawDecodeParams::default();

    let raw_image = decoder.raw_image(&source, &params, false)?;
    let image = RawDevelop::default()
        .develop_intermediate(&raw_image)?
        .to_dynamic_image()
        .ok_or_else(|| anyhow!("develop produced no image"))?;

    let (_, orientation) = exif_extras(decoder.as_ref(), &source, &params);

    Ok(Preview {
        image,
        // the develop pipeline's final step already encodes sRGB
        icc: None,
        orientation,
    })
}

/// The largest embedded image findable: rawler's per-format `full_image`
/// only checks specific IFDs and misses e.g. the full-size JPEG newer Sony
/// bodies store in IFD2 (returning the 1616x1080 preview instead), so the
/// TIFF container is also scanned directly and the bigger result wins
fn largest_embedded(
    decoder: &dyn Decoder,
    source: &RawSource,
    params: &RawDecodeParams,
    path: &Path,
) -> Option<DynamicImage> {
    let from_rawler = decoder.full_image(source, params).ok().flatten();

    let from_scan = largest_tiff_jpeg(path)
        .inspect_err(|e| {
            tracing::debug!("TIFF preview scan failed for {path:?}: {e:?}")
        })
        .ok()
        .flatten()
        .and_then(|blob| {
            image::load_from_memory_with_format(&blob, image::ImageFormat::Jpeg)
                .ok()
        });

    match (from_rawler, from_scan) {
        (Some(a), Some(b)) => {
            Some(match b.width() * b.height() > a.width() * a.height() {
                true => b,
                false => a,
            })
        }
        (a, b) => a.or(b),
    }
}

const TAG_JPEG_OFFSET: u16 = 0x0201; // JPEGInterchangeFormat
const TAG_JPEG_LENGTH: u16 = 0x0202; // JPEGInterchangeFormatLength
const TAG_SUB_IFDS: u16 = 0x014a;

/// Walks the IFD chain (and SubIFDs) of a TIFF-based raw and returns the
/// bytes of the largest embedded JPEG. Non-TIFF containers (CR3, RAF, ...)
/// return None and rawler's own extraction stands.
fn largest_tiff_jpeg(path: &Path) -> Result<Option<Vec<u8>>> {
    let file = File::open(path)?;
    let file_len = file.metadata()?.len();
    let mut reader = BufReader::new(file);

    let mut header = [0u8; 8];
    reader.read_exact(&mut header)?;
    let le = match &header[0..2] {
        b"II" => true,
        b"MM" => false,
        _ => return Ok(None),
    };

    let mut queue = vec![ru32(&header[4..8], le) as u64];
    let mut visited = HashSet::new();
    // (offset, byte length) of the largest jpeg seen
    let mut best: Option<(u64, u64)> = None;

    while let Some(ifd_offset) = queue.pop() {
        if ifd_offset == 0
            || ifd_offset >= file_len
            || !visited.insert(ifd_offset)
            || visited.len() > 64
        {
            continue;
        }

        reader.seek(SeekFrom::Start(ifd_offset))?;
        let mut count_buf = [0u8; 2];
        if reader.read_exact(&mut count_buf).is_err() {
            continue;
        }
        let count = ru16(&count_buf, le) as usize;
        if count == 0 || count > 512 {
            continue;
        }

        let mut entries = vec![0u8; count * 12 + 4];
        if reader.read_exact(&mut entries).is_err() {
            continue;
        }
        // the chain continues after the entry table
        queue.push(ru32(&entries[count * 12..][..4], le) as u64);

        let mut jpeg = (None, None);
        for entry in entries[..count * 12].chunks_exact(12) {
            let tag = ru16(&entry[0..2], le);
            let value_count = ru32(&entry[4..8], le) as u64;
            let value = ru32(&entry[8..12], le) as u64;

            match tag {
                TAG_JPEG_OFFSET => jpeg.0 = Some(value),
                TAG_JPEG_LENGTH => jpeg.1 = Some(value),
                TAG_SUB_IFDS if value_count == 1 => queue.push(value),
                TAG_SUB_IFDS if value_count <= 32 => {
                    // the offset array itself lives out of line
                    let mut offsets = vec![0u8; value_count as usize * 4];
                    if reader.seek(SeekFrom::Start(value)).is_ok()
                        && reader.read_exact(&mut offsets).is_ok()
                    {
                        for chunk in offsets.chunks_exact(4) {
                            queue.push(ru32(chunk, le) as u64);
                        }
                    }
                }
                _ => {}
            }
        }

        if let (Some(offset), Some(len)) = jpeg
            && len > 0
            && offset + len <= file_len
            && best.is_none_or(|(_, best_len)| len > best_len)
        {
            best = Some((offset, len));
        }
    }

    Ok(match best {
        Some((offset, len)) => {
            let mut blob = vec![0u8; len as usize];
            reader.seek(SeekFrom::Start(offset))?;
            reader.read_exact(&mut blob)?;
            Some(blob)
        }
        None => None,
    })
}

fn ru16(buf: &[u8], le: bool) -> u16 {
    let bytes = [buf[0], buf[1]];
    match le {
        true => u16::from_le_bytes(bytes),
        false => u16::from_be_bytes(bytes),
    }
}

fn ru32(buf: &[u8], le: bool) -> u32 {
    let bytes = [buf[0], buf[1], buf[2], buf[3]];
    match le {
        true => u32::from_le_bytes(bytes),
        false => u32::from_be_bytes(bytes),
    }
}

fn exif_extras(
    decoder: &dyn Decoder,
    source: &RawSource,
    params: &RawDecodeParams,
) -> (Option<Vec<u8>>, Option<Orientation>) {
    let meta = decoder.raw_metadata(source, params).ok();

    let orientation = meta
        .as_ref()
        .and_then(|m| m.exif.orientation)
        .and_then(|o| Orientation::from_exif(o.min(255) as u8));

    // EXIF ColorSpace: 1 (or absent) = sRGB, 2 = Adobe RGB
    let icc = match meta.as_ref().and_then(|m| m.exif.color_space) {
        Some(2) => adobe_rgb_icc()
            .inspect_err(|e| {
                tracing::warn!("Failed to build the AdobeRGB profile: {e:?}")
            })
            .ok(),
        _ => None,
    };

    (icc, orientation)
}

/// Adobe RGB (1998), serialized so it can flow through the same ICC path
/// every other embedded profile takes
fn adobe_rgb_icc() -> Result<Vec<u8>> {
    let d65 = lcms2::CIExyY { x: 0.3127, y: 0.3290, Y: 1.0 };
    let primaries = lcms2::CIExyYTRIPLE {
        Red: lcms2::CIExyY { x: 0.6400, y: 0.3300, Y: 1.0 },
        Green: lcms2::CIExyY { x: 0.2100, y: 0.7100, Y: 1.0 },
        Blue: lcms2::CIExyY { x: 0.1500, y: 0.0600, Y: 1.0 },
    };
    let gamma = lcms2::ToneCurve::new(563.0 / 256.0);
    let profile =
        lcms2::Profile::new_rgb(&d65, &primaries, &[&gamma, &gamma, &gamma])?;
    Ok(profile.icc()?)
}
