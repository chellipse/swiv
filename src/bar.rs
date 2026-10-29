//! The single-image mode path bar, rasterized whole on the CPU.
//!
//! The text only changes on navigation or resize, so the bar (background
//! and text) is drawn into one texture exactly its on-screen size and shown
//! as a single pixel-aligned quad through the regular image pipeline. No
//! glyph atlas: that pays off when thousands of glyphs change every frame,
//! not for a hundred that change on a key press.

use std::ops::Range;

use ab_glyph::{
    Font, FontArc, GlyphId, PxScale, PxScaleFont, ScaleFont, point,
};
use wgpu::TextureFormat;

use crate::{
    font::{self, Face},
    lazy::GenericImage,
};

// Sizes in whole pixels at scale factor 1. The height is fixed rather than
// derived from font metrics, so the image area never moves when the font
// arrives, or with a different font. 12px is about the smallest em an
// unhinted rasterizer still draws clean lowercase at.
const BAR_HEIGHT: f64 = 18.0;
const BAR_FONT_EM: f64 = 12.0;
const BAR_PAD_X: f64 = 6.0;

// Authored in sRGB, converted to linear light where they get blended
const BACKGROUND: [u8; 3] = [0x14, 0x17, 0x1c];
const MUTED: [f32; 3] = [0.62, 0.66, 0.72];
const TEXT: [f32; 3] = [0.98, 0.99, 1.0];

const SEP: char = std::path::MAIN_SEPARATOR;

type Color = [f32; 3];

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BarMetrics {
    pub height: u32,
    em_px: f32,
    pad_x: f32,
}

impl BarMetrics {
    /// Each size is rounded to whole physical pixels on its own, so glyphs
    /// keep the same shapes at any scale factor and the bar edge stays on
    /// the pixel grid
    pub fn new(scale_factor: f64) -> Self {
        Self {
            height: (BAR_HEIGHT * scale_factor).round() as u32,
            em_px: (BAR_FONT_EM * scale_factor).round() as f32,
            pad_x: (BAR_PAD_X * scale_factor).round() as f32,
        }
    }
}

/// `faces` is the primary face followed by fallbacks, see
/// [`crate::font::BarFonts::faces`]
pub fn rasterize(
    faces: &[Face],
    metrics: BarMetrics,
    width: u32,
    path: &str,
    position: &str,
) -> GenericImage {
    let height = metrics.height;
    let [r, g, b] = BACKGROUND;
    let mut canvas = Canvas {
        width,
        height,
        bytes: [r, g, b, 255].repeat((width * height) as usize),
    };

    // until the primary font arrives, the bar is just its background
    if !faces.is_empty() {
        draw_text(&mut canvas, faces, metrics, path, position);
    }

    GenericImage {
        width,
        height,
        bytes: canvas.bytes,
        format: TextureFormat::Rgba8UnormSrgb,
        pixel_width: 4,
    }
}

struct Canvas {
    width: u32,
    height: u32,
    bytes: Vec<u8>,
}

impl Canvas {
    /// Blends in linear light, the same way the GPU would
    fn blend(&mut self, x: i32, y: i32, coverage: f32, color: [f32; 3]) {
        if x < 0 || y < 0 || x >= self.width as i32 || y >= self.height as i32 {
            return;
        }
        // lifts antialiased midtones: blended linearly, light-on-dark stem
        // edges otherwise read thin
        let alpha = coverage.clamp(0.0, 1.0).powf(0.75);
        let idx = (y as usize * self.width as usize + x as usize) * 4;
        for (dst, src) in self.bytes[idx..idx + 3].iter_mut().zip(color) {
            let lin = srgb_to_linear(*dst as f32 / 255.0);
            let lin = lin + (srgb_to_linear(src) - lin) * alpha;
            *dst = (linear_to_srgb(lin) * 255.0).round() as u8;
        }
    }
}

fn draw_text(
    canvas: &mut Canvas,
    faces: &[Face],
    metrics: BarMetrics,
    path: &str,
    position: &str,
) {
    let scaled = scale(faces, metrics.em_px);
    let primary = &scaled[0];

    // centres the primary face's ascent-descent box (descent is negative),
    // on a whole pixel so horizontal stems and the x-height stay sharp;
    // fallback glyphs share its baseline
    let baseline =
        ((canvas.height as f32 + primary.ascent() + primary.descent()) / 2.0)
            .round();

    let ellipsis = match faces.iter().any(|face| face.can_draw('…')) {
        true => "…",
        false => "...",
    };

    let position: Vec<_> = shape(&scaled, position.chars())
        .into_iter()
        .map(|glyph| (glyph, MUTED))
        .collect();
    let position_w = walk(&scaled, position.iter().copied(), |_, _, _| {});
    let right = canvas.width as f32 - metrics.pad_x;
    let full_w = right - metrics.pad_x;
    // the position plus a one-em gap comes off the path's space
    let path_w = full_w - position_w - metrics.em_px;

    let chars: Vec<char> = path.chars().collect();
    let path = ShapedPath::new(&scaled, &chars, ellipsis);
    let candidates = elisions(&chars);
    let fits = |avail: f32| {
        candidates.iter().copied().find(|&e| path.width(e) <= avail)
    };

    // a tiny window drops the position before cutting the name any further,
    // and past that the shortest form is clipped at the edge
    let (show_position, elision) = match fits(path_w) {
        Some(e) => (true, e),
        None => (false, fits(full_w).unwrap_or(*candidates.last().unwrap())),
    };

    let mut draw = |x: f32, glyphs: &[(Glyph, Color)]| {
        walk(&scaled, glyphs.iter().copied(), |glyph, pen, color| {
            // origins snap to whole pixels, so repeated letters rasterize
            // identically; the pen itself keeps its fraction
            let face = &scaled[glyph.face];
            let positioned = glyph.id.with_scale_and_position(
                face.scale(),
                point((x + pen).round(), baseline),
            );
            if let Some(outlined) = face.outline_glyph(positioned) {
                let bounds = outlined.px_bounds();
                outlined.draw(|gx, gy, coverage| {
                    canvas.blend(
                        bounds.min.x as i32 + gx as i32,
                        bounds.min.y as i32 + gy as i32,
                        coverage,
                        color,
                    );
                });
            }
        });
    };

    let path_glyphs: Vec<_> = path.glyphs(elision).collect();
    draw(metrics.pad_x, &path_glyphs);
    if show_position {
        draw(right - position_w, &position);
    }
}

fn scale(faces: &[Face], em_px: f32) -> Vec<PxScaleFont<&FontArc>> {
    faces
        .iter()
        .map(|face| {
            face.font
                .as_scaled(PxScale::from(em_px * face.height_per_em))
        })
        .collect()
}

/// A character in the face it's drawn with
#[derive(Debug, Clone, Copy)]
struct Glyph {
    face: usize,
    id: GlyphId,
    advance: f32,
}

/// Each character comes from the first face that can draw it (see
/// [`font::drawable_glyph`]). One that none of them can draws as the
/// primary face's missing-glyph box, until a fallback covering it arrives
fn shape(
    scaled: &[PxScaleFont<&FontArc>],
    text: impl IntoIterator<Item = char>,
) -> Vec<Glyph> {
    text.into_iter()
        .map(|c| {
            let (face, id) = scaled
                .iter()
                .enumerate()
                .find_map(|(idx, face)| {
                    font::drawable_glyph(face.font(), c).map(|id| (idx, id))
                })
                .unwrap_or((0, GlyphId(0)));
            Glyph { face, id, advance: scaled[face].h_advance(id) }
        })
        .collect()
}

/// Kerning only exists between glyphs of the same face
fn kern(scaled: &[PxScaleFont<&FontArc>], a: Glyph, b: Glyph) -> f32 {
    match a.face == b.face {
        true => scaled[a.face].kern(a.id, b.id),
        false => 0.0,
    }
}

/// Lays out glyphs left to right, calling `visit` with each one's pen
/// position; returns the total advance
fn walk(
    scaled: &[PxScaleFont<&FontArc>],
    glyphs: impl IntoIterator<Item = (Glyph, Color)>,
    mut visit: impl FnMut(Glyph, f32, Color),
) -> f32 {
    let mut pen = 0.0;
    let mut prev = None;
    for (glyph, color) in glyphs {
        if let Some(prev) = prev {
            pen += kern(scaled, prev, glyph);
        }
        visit(glyph, pen, color);
        pen += glyph.advance;
        prev = Some(glyph);
    }
    pen
}

/// A way to show the path: its characters from `start` on, with those in
/// `cut` (if any) replaced by an ellipsis
#[derive(Debug, Clone, Copy)]
struct Elision {
    start: usize,
    cut: Option<(usize, usize)>,
}

/// The path shaped and measured once. Every elision is pieces of it around
/// an ellipsis, so its width adds up from running sums rather than another
/// walk over the path: a deep path has as many elisions as directories
struct ShapedPath<'a> {
    scaled: &'a [PxScaleFont<&'a FontArc>],
    glyphs: Vec<Glyph>,
    /// `ends[i]` is the pen after the first `i` glyphs
    ends: Vec<f32>,
    name_start: usize,
    ellipsis: Vec<Glyph>,
    ellipsis_w: f32,
}

impl<'a> ShapedPath<'a> {
    fn new(
        scaled: &'a [PxScaleFont<&'a FontArc>],
        chars: &[char],
        ellipsis: &str,
    ) -> Self {
        let glyphs = shape(scaled, chars.iter().copied());
        let mut ends = vec![0.0];
        walk(
            scaled,
            glyphs.iter().map(|&g| (g, TEXT)),
            |glyph, pen, _| {
                ends.push(pen + glyph.advance);
            },
        );

        let ellipsis = shape(scaled, ellipsis.chars());
        let ellipsis_w =
            walk(scaled, ellipsis.iter().map(|&g| (g, TEXT)), |_, _, _| {});

        Self {
            scaled,
            glyphs,
            ends,
            name_start: name_start(chars),
            ellipsis,
            ellipsis_w,
        }
    }

    /// The same as walking [`Self::glyphs`], without the walk
    fn width(&self, elision: Elision) -> f32 {
        let n = self.glyphs.len();
        let Some((a, b)) = elision.cut else {
            return self.span(elision.start..n);
        };

        let mut width =
            self.span(elision.start..a) + self.ellipsis_w + self.span(b..n);
        // where the ellipsis meets the path
        if let (Some(&before), Some(&first)) =
            (self.glyphs[elision.start..a].last(), self.ellipsis.first())
        {
            width += kern(self.scaled, before, first);
        }
        if let (Some(&last), Some(&after)) =
            (self.ellipsis.last(), self.glyphs.get(b))
        {
            width += kern(self.scaled, last, after);
        }
        width
    }

    /// The width of `range` laid out on its own
    fn span(&self, range: Range<usize>) -> f32 {
        let Range { start, end } = range;
        if start >= end {
            return 0.0;
        }
        // on its own, the first glyph isn't kerned against its neighbour
        let kern_in = match start {
            0 => 0.0,
            _ => kern(self.scaled, self.glyphs[start - 1], self.glyphs[start]),
        };
        self.ends[end] - self.ends[start] - kern_in
    }

    /// Directories are muted, as is an ellipsis standing in for some
    fn glyphs(
        &self,
        elision: Elision,
    ) -> impl Iterator<Item = (Glyph, Color)> + '_ {
        let n = self.glyphs.len();
        let (a, b) = elision.cut.unwrap_or((n, n));
        let ellipsis = match elision.cut {
            Some(_) => &self.ellipsis[..],
            None => &[],
        };
        let color = move |i: usize| match i < self.name_start {
            true => MUTED,
            false => TEXT,
        };
        let path = move |range: Range<usize>| {
            range.map(move |i| (self.glyphs[i], color(i)))
        };

        path(elision.start..a)
            .chain(ellipsis.iter().map(move |&g| (g, color(a))))
            .chain(path(b..n))
    }
}

/// Where the file name starts, after the last separator
fn name_start(path: &[char]) -> usize {
    path.iter()
        .rposition(|&c| c == SEP)
        .map_or(0, |idx| idx + 1)
}

/// Ways to show `path`, longest first:
/// - whole
/// - directories cut from the left at separators (`…/photos/IMG_0001.jpg`)
/// - the file name alone
/// - the file name's stem cut in the middle, keeping the extension
fn elisions(path: &[char]) -> Vec<Elision> {
    let name = name_start(path);

    let mut candidates = vec![Elision { start: 0, cut: None }];

    // the directory's own separators, past a leading one
    candidates.extend(
        (1..name)
            .filter(|&idx| path[idx] == SEP)
            .map(|idx| Elision { start: 0, cut: Some((0, idx)) }),
    );

    if name > 0 {
        candidates.push(Elision { start: name, cut: None });
    }

    // a leading dot is a hidden file's stem, not an extension
    let stem_end = match path[name..].iter().rposition(|&c| c == '.') {
        Some(idx) if idx > 0 => name + idx,
        _ => path.len(),
    };
    candidates.extend((2..stem_end - name).rev().map(|keep| Elision {
        start: name,
        cut: Some((name + keep.div_ceil(2), stem_end - keep / 2)),
    }));

    candidates
}

fn srgb_to_linear(c: f32) -> f32 {
    match c <= 0.04045 {
        true => c / 12.92,
        false => ((c + 0.055) / 1.055).powf(2.4),
    }
}

fn linear_to_srgb(c: f32) -> f32 {
    let c = c.clamp(0.0, 1.0);
    match c <= 0.0031308 {
        true => c * 12.92,
        false => 1.055 * c.powf(1.0 / 2.4) - 0.055,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(path: &str) -> Vec<String> {
        let chars: Vec<char> = path.chars().collect();
        let n = chars.len();
        let text =
            |range: Range<usize>| chars[range].iter().collect::<String>();
        elisions(&chars)
            .into_iter()
            .map(|e| match e.cut {
                Some((a, b)) => format!("{}…{}", text(e.start..a), text(b..n)),
                None => text(e.start..n),
            })
            .collect()
    }

    #[test]
    fn elision_widths_match_their_layout() {
        let faces = [Face::bundled()];
        let scaled = scale(&faces, 12.0);
        let chars: Vec<char> = "/a/photos/2024/IMG_01.jpg".chars().collect();
        let path = ShapedPath::new(&scaled, &chars, "…");
        for e in elisions(&chars) {
            let walked = walk(&scaled, path.glyphs(e), |_, _, _| {});
            assert!((path.width(e) - walked).abs() < 1e-3, "{e:?}");
        }
    }

    #[test]
    fn elides_directories_at_separators_then_the_stem() {
        assert_eq!(
            names("a/photos/2024/IMG_01.jpg"),
            [
                "a/photos/2024/IMG_01.jpg",
                "…/photos/2024/IMG_01.jpg",
                "…/2024/IMG_01.jpg",
                "…/IMG_01.jpg",
                "IMG_01.jpg",
                "IMG…01.jpg",
                "IM…01.jpg",
                "IM…1.jpg",
                "I…1.jpg",
            ]
        );
    }

    #[test]
    fn bare_and_hidden_names() {
        assert_eq!(names("abcd"), ["abcd", "ab…d", "a…d"]);
        assert_eq!(names(".abc"), [".abc", ".a…c", ".…c"]);
        assert_eq!(names("x"), ["x"]);
    }

    #[test]
    fn metrics_round_to_whole_pixels() {
        let m = BarMetrics::new(1.5);
        assert_eq!((m.height, m.em_px, m.pad_x), (27, 18.0, 9.0));
        assert_eq!(BarMetrics::new(1.0).height, 18);
    }
}
