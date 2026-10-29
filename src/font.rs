//! Resolving and loading the path bar's fonts.
//!
//! `--bar-font` takes a family name (real or generic, like `monospace`),
//! which goes to fontconfig, or a path to a font file. The bundled face is
//! the last resort, so the bar still has text where fontconfig is absent.
//!
//! Characters the primary face lacks (CJK in a Latin monospace face, say)
//! get a fallback face from fontconfig, asked for per character as file
//! names turn up needing one.
//!
//! All loading happens on one background thread: a system font can be a
//! 32 MB CJK collection, and `fc-match` alone costs ~10-20ms, none of which
//! belongs on the critical path to the first frame or in a redraw.

use std::{
    cell::RefCell,
    collections::HashSet,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use ab_glyph::{Font, FontArc, FontVec, GlyphId};
use anyhow::{Context, Result};
use crossbeam_channel::{Receiver, Sender, unbounded};

const BUNDLED: &[u8] = include_bytes!("../assets/DejaVuSansMono.ttf");
const BUNDLED_NAME: &str = "bundled DejaVu Sans Mono";

pub const DEFAULT_FAMILY: &str = "monospace";

#[derive(Debug, Clone)]
pub struct Face {
    pub font: FontArc,
    /// Ascent-to-descent height per em.
    ///
    /// ab_glyph's `PxScale` is that height rather than the em, and it varies
    /// a lot between faces: CJK fonts carry tall vertical metrics, so the
    /// same number renders Latin text at about two-thirds the size. Sizes
    /// here mean the em (what CSS means by `font-size`), and this converts
    /// one into the other. It also keeps fallback glyphs the same size as
    /// the primary face's
    pub height_per_em: f32,
}

impl Face {
    fn load(file: &FontFile) -> Result<Self> {
        let bytes = std::fs::read(&file.path)
            .with_context(|| format!("reading font {:?}", file.path))?;
        let font = FontVec::try_from_vec_and_index(bytes, file.index)
            .with_context(|| format!("{file:?} is not a usable font"))?;
        Self::from_font(font.into())
    }

    pub(crate) fn bundled() -> Self {
        let font = FontArc::try_from_slice(BUNDLED).unwrap();
        Self::from_font(font).expect("the bundled font has an em square")
    }

    fn from_font(font: FontArc) -> Result<Self> {
        let units = font
            .units_per_em()
            .context("font declares no em square to measure sizes against")?;
        let height_per_em = font.height_unscaled() / units;
        Ok(Self { font, height_per_em })
    }

    pub fn can_draw(&self, c: char) -> bool {
        drawable_glyph(&self.font, c).is_some()
    }
}

/// The glyph `font` draws `c` with, if it can draw it at all.
///
/// Color emoji fonts map glyphs they only carry as bitmaps (digits too),
/// which ab_glyph can't rasterize; those count as missing, so rendering
/// moves on to the next face and the loader gets asked for one
pub fn drawable_glyph(font: &impl Font, c: char) -> Option<GlyphId> {
    let id = font.glyph_id(c);
    let drawable =
        id.0 != 0 && (c.is_whitespace() || font.outline(id).is_some());
    drawable.then_some(id)
}

/// A face in a font file; collections (.ttc) hold several
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct FontFile {
    path: PathBuf,
    index: u32,
}

/// Faces only ever get appended, so a snapshot's length tells it apart from
/// earlier ones
type SharedFaces = Arc<Mutex<Arc<[Face]>>>;

#[derive(Debug)]
pub struct BarFonts {
    faces: SharedFaces,
    requests: Sender<char>,
    /// sent to the loader already, whether or not a face turned up
    requested: RefCell<HashSet<char>>,
}

impl BarFonts {
    pub fn spawn(spec: String) -> Self {
        let faces: SharedFaces = Arc::new(Mutex::new(Arc::from([])));
        let (requests, receiver) = unbounded();

        let capturable = faces.clone();
        std::thread::spawn(move || load(spec, capturable, receiver));

        Self { faces, requests, requested: RefCell::default() }
    }

    /// The primary face, then fallbacks in the order they arrived; empty
    /// until the primary loads
    pub fn faces(&self) -> Arc<[Face]> {
        self.faces.lock().unwrap().clone()
    }

    /// Asks the loader for faces covering what `faces` lack, once per
    /// character
    pub fn request_missing(&self, faces: &[Face], text: &str) {
        // before the primary arrives, everything would look missing
        if faces.is_empty() {
            return;
        }

        let mut requested = self.requested.borrow_mut();
        for c in text.chars() {
            if c.is_control() || faces.iter().any(|face| face.can_draw(c)) {
                continue;
            }
            if requested.insert(c) {
                let _ = self.requests.send(c);
            }
        }
    }
}

fn push(faces: &SharedFaces, face: Face) {
    let mut faces = faces.lock().unwrap();
    *faces = faces.iter().cloned().chain([face]).collect();
}

fn load(spec: String, faces: SharedFaces, requests: Receiver<char>) {
    let (primary, primary_file) = resolve(&spec);
    push(&faces, primary);

    // keeping the family makes fontconfig prefer covering faces in a
    // similar style: monospace plus katakana gives a mono CJK face
    let family = match looks_like_path(&spec) {
        true => "",
        false => spec.as_str(),
    };

    // in use already, or not a usable font at all; fontconfig answering
    // with one of these means nothing better covers the character. A font
    // that merely lacks one character isn't recorded, as it may cover the
    // next (each character is only requested once, so it's read again at
    // most once per missing character)
    let mut skip: HashSet<FontFile> = primary_file.into_iter().collect();

    for c in requests {
        // a fallback loaded for an earlier character may cover this one
        if faces.lock().unwrap().iter().any(|face| face.can_draw(c)) {
            continue;
        }

        // fontconfig counts bitmap-only glyphs as coverage, so the family
        // can keep answering with a face that can't draw `c` (a color
        // emoji family, for its digits); then any family will do
        let charset = format!(":charset={:x}", c as u32);
        let patterns = [format!("{family}{charset}"), charset];
        let patterns = match family.is_empty() {
            true => &patterns[1..],
            false => &patterns[..],
        };

        // both patterns can name the same font
        let mut lacking = HashSet::new();
        let found = patterns.iter().any(|pattern| {
            let Some(file) = system_font(pattern) else {
                return false;
            };
            if skip.contains(&file) || !lacking.insert(file.clone()) {
                return false;
            }
            match try_fallback(&faces, &file, c) {
                Ok(loaded) => {
                    if loaded {
                        skip.insert(file);
                    }
                    loaded
                }
                Err(e) => {
                    tracing::warn!("{e:#}");
                    skip.insert(file);
                    false
                }
            }
        });
        if !found {
            tracing::debug!("No bar font covers {c:?}");
        }
    }
}

/// Loads `file` as a fallback face if it can draw `c`, saying whether it
/// did; an error means the file is no usable font
fn try_fallback(faces: &SharedFaces, file: &FontFile, c: char) -> Result<bool> {
    let face = Face::load(file)?;
    if !face.can_draw(c) {
        tracing::debug!("Bar fallback font {file:?} can't draw {c:?}");
        return Ok(false);
    }
    tracing::debug!("Bar fallback font for {c:?}: {file:?}");
    push(faces, face);
    Ok(true)
}

/// Whether a `--bar-font` value means a file rather than a family name.
/// Something with a separator or a font extension can only be a path, and
/// a missing one deserves an error rather than fontconfig's idea of a
/// family by that name
pub fn looks_like_path(spec: &str) -> bool {
    Path::new(spec)
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| {
            ["ttf", "otf", "ttc"]
                .iter()
                .any(|k| e.eq_ignore_ascii_case(k))
        })
        || spec.contains(std::path::MAIN_SEPARATOR)
}

/// fontconfig's best match for a pattern, generic family names included
fn system_font(pattern: &str) -> Option<FontFile> {
    let output = std::process::Command::new("fc-match")
        .args(["--format=%{index}\n%{file}", pattern])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let output = String::from_utf8_lossy(&output.stdout);
    let (index, path) = output.split_once('\n')?;
    let path = PathBuf::from(path.trim());
    // the bits above the low 16 pick a variable font's named instance
    // (a weight, say), which isn't applied; the face renders at its
    // default instance
    let index = index.trim().parse::<u32>().unwrap_or(0) & 0xffff;
    path.is_file().then_some(FontFile { path, index })
}

/// The primary face, and the file it came from unless it's the bundled one
fn resolve(spec: &str) -> (Face, Option<FontFile>) {
    let file = match looks_like_path(spec) {
        // NOTE: a .ttc collection given by path loads its first face
        true => Some(FontFile { path: PathBuf::from(spec), index: 0 }),
        false => system_font(spec),
    };

    let Some(file) = file else {
        tracing::warn!(
            "fontconfig resolved no '{spec}' font; using the {BUNDLED_NAME}"
        );
        return (Face::bundled(), None);
    };

    match Face::load(&file) {
        Ok(face) => {
            tracing::debug!("Bar font: {file:?}");
            (face, Some(file))
        }
        Err(e) => {
            tracing::error!("{e:#}; falling back to the {BUNDLED_NAME}");
            (Face::bundled(), None)
        }
    }
}
