//! Minimal PSD/PSB reader: parses only what a viewer needs — the flattened
//! composite Photoshop stores at the end of the file, present whenever the
//! file was saved with "Maximize Compatibility" (the default). Layers are
//! never parsed. PSB ("large document format") is identical to PSD except
//! for a version marker and a few widened length fields, so one reader
//! covers both.
//!
//! Reference: Adobe Photoshop File Format Specification.

use std::{
    fs::File,
    io::{BufReader, Read, Seek, SeekFrom},
    path::Path,
};

use anyhow::{Result, anyhow, bail};

const COLOR_MODE_GRAYSCALE: u16 = 1;
const COLOR_MODE_RGB: u16 = 3;

const ICC_RESOURCE_ID: u16 = 1039;

pub struct PsdComposite {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
    /// Only captured for RGB documents; the lcms2 transform downstream
    /// assumes an RGB-shaped profile
    pub icc: Option<Vec<u8>>,
}

struct Header {
    is_psb: bool,
    channels: u16,
    width: u32,
    height: u32,
    depth: u16,
    color_mode: u16,
}

pub fn probe_dimensions(path: &Path) -> Result<(u32, u32)> {
    let mut reader = BufReader::new(File::open(path)?);
    let header = read_header(&mut reader)?;
    Ok((header.width, header.height))
}

pub fn decode(path: &Path, max_alloc: Option<u64>) -> Result<PsdComposite> {
    let mut reader = BufReader::new(File::open(path)?);
    let header = read_header(&mut reader)?;

    match (header.color_mode, header.depth) {
        (COLOR_MODE_GRAYSCALE | COLOR_MODE_RGB, 8 | 16) => {}
        (mode, depth) => {
            bail!("unsupported psd color mode {mode} at depth {depth}")
        }
    }

    let color_channels = match header.color_mode {
        COLOR_MODE_GRAYSCALE => 1usize,
        _ => 3,
    };
    if (header.channels as usize) < color_channels {
        bail!(
            "psd reports {} channels but its color mode needs {color_channels}",
            header.channels
        );
    }

    // Extra channels beyond the color data: the first is treated as
    // transparency. (For a file carrying a saved selection instead this is
    // wrong, but the composite is matted during flattening, so opaque
    // rendering output is unaffected either way.)
    let has_alpha = (header.channels as usize) > color_channels;
    let used_channels = color_channels + has_alpha as usize;

    let width = header.width as usize;
    let height = header.height as usize;
    let bytes_per_sample = (header.depth / 8) as usize;

    if let Some(max) = max_alloc {
        let planes = (width * height * bytes_per_sample * used_channels) as u64;
        let rgba = (width * height * 4) as u64;
        if planes + rgba > max {
            bail!(
                "psd decode would allocate {} MB, over the decode budget",
                (planes + rgba) / 1_000_000
            );
        }
    }

    // color mode data (palettes etc.) — empty for grayscale/rgb
    let len = read_u32(&mut reader)?;
    skip(&mut reader, len as u64)?;

    let icc = scan_image_resources(&mut reader)?
        .filter(|_| header.color_mode == COLOR_MODE_RGB);

    // layer and mask info — never parsed, the composite is all we want
    let len = match header.is_psb {
        true => read_u64(&mut reader)?,
        false => read_u32(&mut reader)? as u64,
    };
    skip(&mut reader, len)?;

    let row_bytes = width * bytes_per_sample;
    let compression = read_u16(&mut reader)?;
    let planes = match compression {
        0 => read_raw_planes(&mut reader, used_channels, height * row_bytes)?,
        1 => read_rle_planes(&mut reader, &header, used_channels, row_bytes)?,
        2 | 3 => bail!("zip-compressed psd composite not supported"),
        x => bail!("unknown psd compression method {x}"),
    };

    // interleave the planar big-endian samples into rgba8, taking the high
    // byte of 16-bit samples
    let mut rgba = vec![0u8; width * height * 4];
    let sample = |plane: &[u8], i: usize| plane[i * bytes_per_sample];
    for i in 0..width * height {
        let [r, g, b] = match color_channels {
            1 => [sample(&planes[0], i); 3],
            _ => [
                sample(&planes[0], i),
                sample(&planes[1], i),
                sample(&planes[2], i),
            ],
        };
        let a = match has_alpha {
            true => sample(&planes[color_channels], i),
            false => u8::MAX,
        };
        rgba[i * 4..i * 4 + 4].copy_from_slice(&[r, g, b, a]);
    }

    Ok(PsdComposite { width: header.width, height: header.height, rgba, icc })
}

fn read_header(reader: &mut impl Read) -> Result<Header> {
    let mut buf = [0u8; 26];
    reader.read_exact(&mut buf)?;

    if &buf[0..4] != b"8BPS" {
        bail!("missing 8BPS signature");
    }

    let is_psb = match u16::from_be_bytes([buf[4], buf[5]]) {
        1 => false,
        2 => true,
        v => bail!("unsupported psd version {v}"),
    };

    Ok(Header {
        is_psb,
        channels: u16::from_be_bytes([buf[12], buf[13]]),
        height: u32::from_be_bytes(buf[14..18].try_into().unwrap()),
        width: u32::from_be_bytes(buf[18..22].try_into().unwrap()),
        depth: u16::from_be_bytes([buf[22], buf[23]]),
        color_mode: u16::from_be_bytes([buf[24], buf[25]]),
    })
}

/// Walks the image resources section, returning the ICC profile
/// (resource 1039) if present. Always leaves the reader at the end of the
/// section.
fn scan_image_resources(
    reader: &mut BufReader<File>,
) -> Result<Option<Vec<u8>>> {
    let section_len = read_u32(reader)? as u64;
    let section_end = reader.stream_position()? + section_len;

    let mut icc = None;
    // 12 = minimum block size: signature + id + empty name + data length
    while reader.stream_position()? + 12 <= section_end {
        let mut signature = [0u8; 4];
        reader.read_exact(&mut signature)?;
        if &signature != b"8BIM" {
            // unknown block signature; abandon the scan rather than walk
            // garbage
            break;
        }

        let id = read_u16(reader)?;

        // pascal string name, padded so the length byte + name total an
        // even byte count
        let name_len = read_u8(reader)? as u64;
        skip(reader, name_len + (name_len + 1) % 2)?;

        let size = read_u32(reader)? as u64;
        if id == ICC_RESOURCE_ID && icc.is_none() {
            let mut data = vec![0u8; size as usize];
            reader.read_exact(&mut data)?;
            icc = Some(data);
        } else {
            skip(reader, size)?;
        }
        // resource data is padded to even length
        skip(reader, size % 2)?;
    }

    reader.seek(SeekFrom::Start(section_end))?;
    Ok(icc)
}

fn read_raw_planes(
    reader: &mut impl Read,
    used_channels: usize,
    plane_bytes: usize,
) -> Result<Vec<Vec<u8>>> {
    let mut planes = Vec::with_capacity(used_channels);
    for _ in 0..used_channels {
        let mut plane = vec![0u8; plane_bytes];
        reader.read_exact(&mut plane)?;
        planes.push(plane);
    }
    Ok(planes)
}

fn read_rle_planes(
    reader: &mut impl Read,
    header: &Header,
    used_channels: usize,
    row_bytes: usize,
) -> Result<Vec<Vec<u8>>> {
    let rows = header.height as usize;

    // the byte-count table up front covers every channel's rows, including
    // channels we won't read
    let total_rows = header.channels as usize * rows;
    let mut counts = Vec::with_capacity(total_rows);
    for _ in 0..total_rows {
        counts.push(match header.is_psb {
            true => read_u32(reader)? as usize,
            false => read_u16(reader)? as usize,
        });
    }

    let mut planes = Vec::with_capacity(used_channels);
    let mut packed = Vec::new();
    for channel in 0..used_channels {
        let mut plane = Vec::with_capacity(rows * row_bytes);
        for row in 0..rows {
            packed.resize(counts[channel * rows + row], 0);
            reader.read_exact(&mut packed)?;
            unpack_bits(&packed, row_bytes, &mut plane)?;
        }
        planes.push(plane);
    }
    Ok(planes)
}

/// PackBits decompression of one row, appending exactly `expected` bytes
/// to `out`
fn unpack_bits(
    packed: &[u8],
    expected: usize,
    out: &mut Vec<u8>,
) -> Result<()> {
    let start = out.len();
    let mut i = 0;
    while out.len() - start < expected {
        let control = *packed
            .get(i)
            .ok_or_else(|| anyhow!("truncated packbits row"))?
            as i8;
        i += 1;

        match control {
            -128 => {} // no-op
            0.. => {
                let count = control as usize + 1;
                let literal = packed
                    .get(i..i + count)
                    .ok_or_else(|| anyhow!("truncated packbits literal"))?;
                out.extend_from_slice(literal);
                i += count;
            }
            _ => {
                let count = (1 - control as i64) as usize;
                let byte = *packed
                    .get(i)
                    .ok_or_else(|| anyhow!("truncated packbits run"))?;
                i += 1;
                out.extend(std::iter::repeat_n(byte, count));
            }
        }
    }

    if out.len() - start != expected {
        bail!("packbits row decoded past its expected length");
    }
    Ok(())
}

fn read_u8(reader: &mut impl Read) -> Result<u8> {
    let mut buf = [0u8; 1];
    reader.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn read_u16(reader: &mut impl Read) -> Result<u16> {
    let mut buf = [0u8; 2];
    reader.read_exact(&mut buf)?;
    Ok(u16::from_be_bytes(buf))
}

fn read_u32(reader: &mut impl Read) -> Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_be_bytes(buf))
}

fn read_u64(reader: &mut impl Read) -> Result<u64> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)?;
    Ok(u64::from_be_bytes(buf))
}

fn skip(reader: &mut BufReader<File>, n: u64) -> Result<()> {
    reader.seek_relative(n as i64)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn header(
        version: u16,
        channels: u16,
        w: u32,
        h: u32,
        depth: u16,
        mode: u16,
    ) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"8BPS");
        buf.extend_from_slice(&version.to_be_bytes());
        buf.extend_from_slice(&[0u8; 6]);
        buf.extend_from_slice(&channels.to_be_bytes());
        buf.extend_from_slice(&h.to_be_bytes());
        buf.extend_from_slice(&w.to_be_bytes());
        buf.extend_from_slice(&depth.to_be_bytes());
        buf.extend_from_slice(&mode.to_be_bytes());
        buf
    }

    fn write_temp(name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let path = std::env::temp_dir()
            .join(format!("swiv-psd-test-{name}-{}", std::process::id()));
        File::create(&path).unwrap().write_all(bytes).unwrap();
        path
    }

    #[test]
    fn raw_rgb_psd() {
        let mut file = header(1, 3, 2, 2, 8, COLOR_MODE_RGB);
        file.extend_from_slice(&0u32.to_be_bytes()); // color mode data
        file.extend_from_slice(&0u32.to_be_bytes()); // image resources
        file.extend_from_slice(&0u32.to_be_bytes()); // layers
        file.extend_from_slice(&0u16.to_be_bytes()); // compression: raw
        file.extend_from_slice(&[1, 2, 3, 4]); // r plane
        file.extend_from_slice(&[5, 6, 7, 8]); // g plane
        file.extend_from_slice(&[9, 10, 11, 12]); // b plane

        let path = write_temp("raw-rgb", &file);
        let composite = decode(&path, None).unwrap();
        std::fs::remove_file(&path).unwrap();

        assert_eq!((composite.width, composite.height), (2, 2));
        assert_eq!(
            composite.rgba,
            [1, 5, 9, 255, 2, 6, 10, 255, 3, 7, 11, 255, 4, 8, 12, 255]
        );
    }

    #[test]
    fn rle_rgba_psd() {
        let mut file = header(1, 4, 2, 1, 8, COLOR_MODE_RGB);
        file.extend_from_slice(&0u32.to_be_bytes());
        file.extend_from_slice(&0u32.to_be_bytes());
        file.extend_from_slice(&0u32.to_be_bytes());
        file.extend_from_slice(&1u16.to_be_bytes()); // compression: rle
        // 4 channels x 1 row of u16 byte counts
        for _ in 0..4 {
            file.extend_from_slice(&3u16.to_be_bytes());
        }
        // each row: literal control byte for 2 bytes
        file.extend_from_slice(&[1, 10, 11]); // r
        file.extend_from_slice(&[1, 20, 21]); // g
        file.extend_from_slice(&[1, 30, 31]); // b
        file.extend_from_slice(&[1, 40, 41]); // a

        let path = write_temp("rle-rgba", &file);
        let composite = decode(&path, None).unwrap();
        std::fs::remove_file(&path).unwrap();

        assert_eq!(composite.rgba, [10, 20, 30, 40, 11, 21, 31, 41]);
    }

    #[test]
    fn raw_gray16_psb() {
        let mut file = header(2, 1, 1, 2, 16, COLOR_MODE_GRAYSCALE);
        file.extend_from_slice(&0u32.to_be_bytes());
        file.extend_from_slice(&0u32.to_be_bytes());
        file.extend_from_slice(&0u64.to_be_bytes()); // layers length is u64 in psb
        file.extend_from_slice(&0u16.to_be_bytes()); // compression: raw
        file.extend_from_slice(&0xabcdu16.to_be_bytes());
        file.extend_from_slice(&0x1234u16.to_be_bytes());

        let path = write_temp("gray16-psb", &file);
        let composite = decode(&path, None).unwrap();
        std::fs::remove_file(&path).unwrap();

        // high bytes of the 16-bit samples, replicated across rgb
        assert_eq!(
            composite.rgba,
            [0xab, 0xab, 0xab, 255, 0x12, 0x12, 0x12, 255]
        );
    }

    #[test]
    fn rle_row_counts_are_u32_in_psb() {
        let mut file = header(2, 1, 2, 1, 8, COLOR_MODE_GRAYSCALE);
        file.extend_from_slice(&0u32.to_be_bytes());
        file.extend_from_slice(&0u32.to_be_bytes());
        file.extend_from_slice(&0u64.to_be_bytes());
        file.extend_from_slice(&1u16.to_be_bytes());
        file.extend_from_slice(&2u32.to_be_bytes()); // one row's byte count, u32
        file.extend_from_slice(&[0xff, 42]); // run: repeat 42 twice (control -1)

        let path = write_temp("rle-psb", &file);
        let composite = decode(&path, None).unwrap();
        std::fs::remove_file(&path).unwrap();

        assert_eq!(composite.rgba, [42, 42, 42, 255, 42, 42, 42, 255]);
    }
}
