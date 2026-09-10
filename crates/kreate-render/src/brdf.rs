//! The standard surface: one BRDF, for the viewport and the renderer both.
//!
//! §2.3's rule, and §5's second risk is why it is a rule rather than a
//! preference: two shading implementations drift, the user reports "the
//! viewport and the render disagree", and that report recurs for as long as
//! both exist. So there is one of these, mirrored in WGSL, and a device test
//! holds the two together.
//!
//! # Metal-roughness, because two formats already chose it
//!
//! glTF's `pbrMetallicRoughness` and `UsdPreviewSurface` are nearly the same
//! parameterisation, and §5.6 noted they were designed to match each other. A
//! specular-glossiness model would need converting at both boundaries, and the
//! conversion is not invertible.
//!
//! # Multiple scattering is compensated from the first day
//!
//! This is the part that could not be deferred. A microfacet model traces one
//! bounce off the microsurface, so the light that would have bounced again is
//! simply lost — and the loss grows with roughness. A rough metal comes out
//! **visibly darker than it should be**, by a third or more at high roughness.
//!
//! Adding the compensation later is not a bug fix, it is a change to the
//! appearance of every material anybody has authored: every rough metal in
//! every saved scene gets brighter, and the values an artist chose to
//! compensate become wrong. §2.3 says so, and it is the reason this file has an
//! energy-conservation test rather than a "looks right" one.
//!
//! The compensation is Turquin's, in the form Fdez-Agüera's paper uses: scale
//! the single-scattering lobe by `1 + F0·(1/E − 1)`, where `E` is that lobe's
//! directional albedo with the Fresnel term held at one. For a white metal that
//! is exactly `1/E`, so the total comes back to one — which is what the furnace
//! test measures.
//!
//! # What is here and what is not
//!
//! Two lobes: Lambert diffuse and GGX specular. Clearcoat, sheen, transmission
//! and subsurface are named in §2.3 and are additional lobes on top of these;
//! none of them changes the two below, which is why starting with two is not a
//! decision that has to be revisited.
//!
//! Not here: the cosine term, the light's colour, or any illuminance.
//! [`evaluate`] answers with the BRDF itself, per steradian, so that it can be
//! integrated — which is what makes the energy claims checkable at all.
//! [`shade`] is the product a fragment wants.

use kreate_math::SceneRgb;

/// A surface, as glTF and `UsdPreviewSurface` both describe one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Surface {
    /// The base colour: a reflectance, bounded by one.
    ///
    /// For a dielectric this is the diffuse albedo; for a metal it is the
    /// specular colour at normal incidence. That one field means two things is
    /// the metal-roughness parameterisation's whole trick, and `metallic` is
    /// what says which.
    pub base_color: SceneRgb,
    /// Zero for a dielectric, one for a metal. Values in between are not a
    /// physical material — they are a texture's transition between two, which
    /// is the only reason the parameter is continuous.
    pub metallic: f32,
    /// Perceptual roughness, zero to one.
    ///
    /// *Perceptual*: the microfacet distribution uses `roughness²`, and the
    /// squaring is what makes the slider feel even. Storing the squared value
    /// would crowd every interesting appearance into the bottom fifth of the
    /// range.
    pub roughness: f32,
}

impl Default for Surface {
    fn default() -> Self {
        Self {
            // A mid grey dielectric: the material an object should arrive
            // with, being the one that shows shape without deciding anything.
            base_color: SceneRgb::new(0.8, 0.8, 0.8),
            metallic: 0.0,
            roughness: 0.5,
        }
    }
}

/// The reflectance of a dielectric at normal incidence.
///
/// Four percent, which is glass, most plastics, and near enough everything
/// that is not a metal. A material parameter for it exists in the full
/// standard surface and is not here yet, because the number is right for
/// almost everything and wrong in ways nobody has asked about.
pub const DIELECTRIC_F0: f32 = 0.04;

impl Surface {
    /// Reflectance at normal incidence.
    ///
    /// A dielectric's four percent, or a metal's own colour — which is what
    /// makes gold gold. Blended by `metallic` because a texture's transition
    /// between the two has to be continuous.
    pub fn f0(&self) -> [f32; 3] {
        let m = self.metallic.clamp(0.0, 1.0);
        let base = self.base_color.to_array();
        [
            DIELECTRIC_F0 * (1.0 - m) + base[0] * m,
            DIELECTRIC_F0 * (1.0 - m) + base[1] * m,
            DIELECTRIC_F0 * (1.0 - m) + base[2] * m,
        ]
    }

    /// The diffuse albedo.
    ///
    /// Zero for a metal, and that is not an approximation: a conductor's
    /// electrons absorb what enters it, so there is no subsurface scattering to
    /// come back out. A metal with a diffuse term is the commonest way for a
    /// renderer to make metal look like painted plastic.
    pub fn diffuse_albedo(&self) -> [f32; 3] {
        let m = 1.0 - self.metallic.clamp(0.0, 1.0);
        let base = self.base_color.to_array();
        [base[0] * m, base[1] * m, base[2] * m]
    }

    /// The microfacet distribution's width.
    ///
    /// `roughness²`, and clamped away from zero: a perfectly smooth surface is
    /// a delta function, which no finite sample of a punctual light can
    /// represent — it would be either invisible or an infinity depending on
    /// where the sample fell. The floor is small enough to look mirror-smooth.
    pub fn alpha(&self) -> f32 {
        let r = self.roughness.clamp(MIN_ROUGHNESS, 1.0);
        r * r
    }
}

/// The smallest roughness the model will use.
///
/// Not zero. A zero-roughness GGX lobe is a delta function: with a punctual
/// light the highlight is a single direction, so it is either missed entirely
/// or lands as an unbounded value. Every real renderer clamps, and saying so
/// here is better than each pass choosing its own floor.
pub const MIN_ROUGHNESS: f32 = 0.045;

/// The BRDF's two lobes, per steradian.
///
/// Kept apart because a light can be told to affect one and not the other
/// (§2.2's lighting cheats), and because the diffuse half is what an
/// irradiance probe supplies while the specular half needs a reflection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Response {
    pub diffuse: [f32; 3],
    pub specular: [f32; 3],
}

impl Response {
    pub const ZERO: Self = Self {
        diffuse: [0.0; 3],
        specular: [0.0; 3],
    };

    /// Both lobes together.
    pub fn total(&self) -> [f32; 3] {
        [
            self.diffuse[0] + self.specular[0],
            self.diffuse[1] + self.specular[1],
            self.diffuse[2] + self.specular[2],
        ]
    }
}

/// GGX's normal distribution: Trowbridge-Reitz.
///
/// Chosen over Beckmann for its tail, which is what makes a rough metal look
/// like metal rather than like chalk.
fn distribution(noh: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let d = noh * noh * (a2 - 1.0) + 1.0;
    // The division is by `d²`, which vanishes only if `a2` is zero — which
    // `MIN_ROUGHNESS` forbids.
    a2 / (std::f32::consts::PI * d * d).max(f32::MIN_POSITIVE)
}

/// Smith's height-correlated visibility term, with the `1/(4·NoL·NoV)` of the
/// microfacet denominator folded in.
///
/// Height-correlated rather than separable: the separable form is brighter at
/// grazing angles by enough to show as a rim that is not there.
fn visibility(nov: f32, nol: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let v = nol * (nov * nov * (1.0 - a2) + a2).sqrt();
    let l = nov * (nol * nol * (1.0 - a2) + a2).sqrt();
    0.5 / (v + l).max(f32::MIN_POSITIVE)
}

/// Schlick's approximation to the Fresnel term.
fn fresnel(f0: [f32; 3], voh: f32) -> [f32; 3] {
    // `(1 - cos)^5`, by multiplication rather than `powf`: five multiplies
    // against a transcendental, per light per fragment.
    let m = (1.0 - voh).clamp(0.0, 1.0);
    let m2 = m * m;
    let f = m2 * m2 * m;
    [
        f0[0] + (1.0 - f0[0]) * f,
        f0[1] + (1.0 - f0[1]) * f,
        f0[2] + (1.0 - f0[2]) * f,
    ]
}

/// How finely the directional albedo is tabulated, per axis.
///
/// Thirty-two squared is a thousand entries, and the function is smooth enough
/// in both arguments that interpolating between them stays well inside the
/// furnace test's tolerance. It is also small enough to hand a shader as a
/// buffer rather than as a texture with a sampler.
pub const ALBEDO_LUT: usize = 32;

/// The GGX lobe's directional albedo, tabulated — the DFG table.
///
/// Two numbers per cell, `(a, b)`, for which the single-scattering albedo of a
/// surface with reflectance `F0` is `F0·a + b`. So `a + b` is the albedo with
/// Fresnel held at one, which the energy compensation needs, and `F0·a + b` is
/// what a particular material actually reflects, which the diffuse lobe needs
/// in order not to reflect it a second time.
///
/// # Why two channels and not one
///
/// A single channel gives `a + b` and nothing else, and that is not enough. The
/// diffuse and specular lobes have to be coupled by *how much the specular lobe
/// actually took* — a directional-hemispherical quantity — and that depends on
/// `F0`, so the two parts have to be separable.
///
/// The first attempt coupled them per light direction instead, scaling the
/// diffuse by `1 − F(VoH)`. It reads as reasonable and glTF's own text is
/// written that way, but it does not conserve energy: at a grazing view a
/// dielectric's specular albedo rises to forty percent — which is physically
/// right, and is why a shallow view of water is mirror-like — while `F(VoH)` is
/// small for most of the light directions, so the diffuse lobe barely gave
/// anything up. A white plastic came out reflecting a third more than arrived.
/// The energy test caught it.
///
/// This is also exactly the table §2.2's IBL prefilter needs, which is the
/// other reason not to economise on it here.
///
/// # Why a table and not a formula
///
/// The first attempt used Lazarov's analytic split-sum fit, on the reasoning
/// that its `scale + bias` is the albedo at `F0 = 1`. It is not. The two halves
/// of that fit are `∓1.04·a + c` and `+1.04·a + d`, so the view-dependent term
/// **cancels exactly** and what remains is `1 − 0.55·roughness`: a straight
/// line with no dependence on the view angle at all.
///
/// The true albedo varies by a factor of two across view angles at high
/// roughness. Using the line over-brightened a smooth metal by six percent and
/// got a rough one wrong in the other direction. The furnace test caught it,
/// which is what it is for — and it is worth recording that the fit is not
/// wrong, it was made for a different job: multiplying by `F0` before the two
/// halves are summed, which is exactly what makes the cancellation harmless
/// there and fatal here.
///
/// So the table is integrated, by importance sampling the same distribution the
/// BRDF uses. That makes the compensation correct by construction rather than
/// by a fit somebody has to trust, and it is the same table §2.2's IBL
/// prefilter needs — arriving at it twice would be the more expensive choice.
pub struct AlbedoTable {
    /// Row-major in roughness: a row is a sweep across view angles.
    entries: [[[f32; 2]; ALBEDO_LUT]; ALBEDO_LUT],
    /// The cosine-weighted average of each row: `(a, b)` averaged over the
    /// hemisphere, per roughness.
    ///
    /// What the coupled diffuse lobe divides by. Reduced from the table above
    /// rather than integrated again, so the two cannot disagree.
    average: [[f32; 2]; ALBEDO_LUT],
}

/// The van der Corput radical inverse, base two.
///
/// Half of a Hammersley sequence. Deterministic, so the table is the same on
/// every machine and these tests are not testing a random number generator.
fn radical_inverse(mut bits: u32) -> f32 {
    bits = bits.rotate_left(16);
    bits = ((bits & 0x5555_5555) << 1) | ((bits & 0xAAAA_AAAA) >> 1);
    bits = ((bits & 0x3333_3333) << 2) | ((bits & 0xCCCC_CCCC) >> 2);
    bits = ((bits & 0x0F0F_0F0F) << 4) | ((bits & 0xF0F0_F0F0) >> 4);
    bits = ((bits & 0x00FF_00FF) << 8) | ((bits & 0xFF00_FF00) >> 8);
    bits as f32 * 2.328_306_4e-10
}

impl AlbedoTable {
    /// Integrates the table. Half a million samples, once.
    pub fn compute() -> Self {
        const SAMPLES: u32 = 512;
        let mut entries = [[[0.0f32; 2]; ALBEDO_LUT]; ALBEDO_LUT];

        for (ri, row) in entries.iter_mut().enumerate() {
            // Cell centres, so no entry sits exactly on roughness zero — where
            // the lobe is a delta function and the estimator has nothing to
            // average.
            let roughness = ((ri as f32 + 0.5) / ALBEDO_LUT as f32).clamp(MIN_ROUGHNESS, 1.0);
            let alpha = roughness * roughness;
            for (vi, cell) in row.iter_mut().enumerate() {
                let nov = ((vi as f32 + 0.5) / ALBEDO_LUT as f32).clamp(1e-3, 1.0);
                let v = glam::Vec3::new((1.0 - nov * nov).max(0.0).sqrt(), 0.0, nov);

                let (mut sum_a, mut sum_b) = (0.0f32, 0.0f32);
                for i in 0..SAMPLES {
                    let u1 = (i as f32 + 0.5) / SAMPLES as f32;
                    let u2 = radical_inverse(i);
                    // A half-vector from the GGX distribution.
                    let phi = std::f32::consts::TAU * u1;
                    let cos_theta = ((1.0 - u2) / (1.0 + (alpha * alpha - 1.0) * u2))
                        .max(0.0)
                        .sqrt();
                    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
                    let h =
                        glam::Vec3::new(sin_theta * phi.cos(), sin_theta * phi.sin(), cos_theta);
                    let l = 2.0 * v.dot(h) * h - v;
                    if l.z <= 0.0 {
                        // Reflected below the horizon: masked, and genuinely
                        // lost. Skipping rather than clamping is the whole
                        // reason the albedo falls off at grazing angles.
                        continue;
                    }
                    let nol = l.z;
                    let noh = h.z.max(1e-6);
                    let voh = v.dot(h).max(1e-6);
                    // The estimator for the integral of `D·V·NoL`: sampling from
                    // the distribution cancels `D`, leaving the visibility term
                    // and the change of variables from the half-vector to the
                    // light direction.
                    let g = 4.0 * visibility(nov, nol, alpha) * nol * voh / noh;
                    // Schlick split into the part that scales with `F0` and the
                    // part that does not, so that `F0·a + b` reconstructs the
                    // Fresnel-weighted albedo for any material.
                    let m = (1.0 - voh).clamp(0.0, 1.0);
                    let m2 = m * m;
                    let fc = m2 * m2 * m;
                    sum_a += (1.0 - fc) * g;
                    sum_b += fc * g;
                }
                let n = SAMPLES as f32;
                let (a, b) = ((sum_a / n).max(0.0), (sum_b / n).max(0.0));
                // The *sum* is the physical bound, not each channel: the albedo
                // at F0 = 1 is `a + b` and cannot exceed one. The estimator's
                // variance pushes it a hair over near a mirror, and everything
                // downstream divides by `1 - E`, so the bound is normalised
                // here rather than clamped at each use.
                let total = a + b;
                let scale = if total > 1.0 { 1.0 / total } else { 1.0 };
                *cell = [a * scale, b * scale];
            }
        }

        // The cosine-weighted average of each row. `∫ cos dω = π` over the
        // hemisphere, and the cells are evenly spaced in `NoV = cosθ`, so a
        // cosine weighting is just `2·NoV` per cell — which is the density that
        // integrates to one.
        let mut average = [[0.0f32; 2]; ALBEDO_LUT];
        for (ri, row) in entries.iter().enumerate() {
            let (mut a, mut b) = (0.0f32, 0.0f32);
            for (vi, cell) in row.iter().enumerate() {
                let nov = (vi as f32 + 0.5) / ALBEDO_LUT as f32;
                let weight = 2.0 * nov / ALBEDO_LUT as f32;
                a += cell[0] * weight;
                b += cell[1] * weight;
            }
            average[ri] = [a, b];
        }

        Self { entries, average }
    }

    /// `(a, b)` at `(NoV, roughness)`, interpolated between cells.
    ///
    /// The single-scattering albedo of a material with reflectance `F0` is
    /// `F0·a + b`; with Fresnel at one it is `a + b`.
    pub fn lookup(&self, nov: f32, roughness: f32) -> (f32, f32) {
        let last = (ALBEDO_LUT - 1) as f32;
        let to_index =
            |t: f32| (t.clamp(0.0, 1.0) * ALBEDO_LUT as f32 - 0.5).clamp(0.0, last);
        let (x, y) = (to_index(nov), to_index(roughness));
        let (x0, y0) = (x.floor() as usize, y.floor() as usize);
        let (x1, y1) = ((x0 + 1).min(ALBEDO_LUT - 1), (y0 + 1).min(ALBEDO_LUT - 1));
        let (fx, fy) = (x - x0 as f32, y - y0 as f32);
        let mix = |c: usize| {
            let near = self.entries[y0][x0][c] * (1.0 - fx) + self.entries[y0][x1][c] * fx;
            let far = self.entries[y1][x0][c] * (1.0 - fx) + self.entries[y1][x1][c] * fx;
            near * (1.0 - fy) + far * fy
        };
        (mix(0), mix(1))
    }

    /// The cosine-weighted average of `(a, b)` over the hemisphere, at
    /// `roughness`.
    ///
    /// What the coupled diffuse lobe divides by, so that its own hemispherical
    /// integral comes out at exactly the energy the specular lobe left.
    pub fn average(&self, roughness: f32) -> (f32, f32) {
        let last = (ALBEDO_LUT - 1) as f32;
        let y = (roughness.clamp(0.0, 1.0) * ALBEDO_LUT as f32 - 0.5).clamp(0.0, last);
        let y0 = y.floor() as usize;
        let y1 = (y0 + 1).min(ALBEDO_LUT - 1);
        let fy = y - y0 as f32;
        (
            self.average[y0][0] * (1.0 - fy) + self.average[y1][0] * fy,
            self.average[y0][1] * (1.0 - fy) + self.average[y1][1] * fy,
        )
    }

    /// The averages as a flat array, for a shader.
    pub fn averages_flat(&self) -> Vec<f32> {
        self.average
            .iter()
            .flat_map(|pair| pair.iter().copied())
            .collect()
    }

    /// The table as a flat array of `(a, b)` pairs, for a shader.
    ///
    /// Row-major in roughness, interleaved, which is what a `vec2` array in
    /// WGSL reads and what a two-channel texture would hold.
    pub fn as_flat(&self) -> Vec<f32> {
        self.entries
            .iter()
            .flatten()
            .flat_map(|pair| pair.iter().copied())
            .collect()
    }
}

/// The one table.
///
/// A mathematical constant with no inputs, so a lazily computed singleton is
/// what it is. Recomputing it per evaluation would be absurd, and threading it
/// through every call would put a parameter on the BRDF with one possible
/// value.
pub fn albedo_table() -> &'static AlbedoTable {
    static TABLE: std::sync::OnceLock<AlbedoTable> = std::sync::OnceLock::new();
    TABLE.get_or_init(AlbedoTable::compute)
}

/// How much the specular lobe must be scaled to put back what one bounce lost.
///
/// Turquin's compensation: `1 + F0·(1/E − 1)`, where `E` is the lobe's
/// directional albedo with Fresnel held at one. For a white metal that is
/// `1/E`, so the total energy returns to one — which is what the furnace test
/// measures, and what "the compensation cannot be added later" means.
///
/// Scaled by `F0` because that is how much light was available to bounce again:
/// a dielectric reflecting four percent has almost nothing to put back, and a
/// mirror has all of it.
fn energy_compensation(f0: [f32; 3], nov: f32, roughness: f32) -> [f32; 3] {
    let (a, b) = albedo_table().lookup(nov, roughness);
    let e = (a + b).clamp(1e-2, 1.0);
    let gain = 1.0 / e - 1.0;
    [1.0 + f0[0] * gain, 1.0 + f0[1] * gain, 1.0 + f0[2] * gain]
}

/// How much of the arriving light the specular lobe takes, per channel.
///
/// `F0·a + b` from the table, times the multiple-scattering compensation — so
/// it accounts for the light that bounces more than once, which is the whole
/// point of the compensation. What is left is what the diffuse lobe may have.
fn specular_share(f0: [f32; 3], nov: f32, roughness: f32) -> [f32; 3] {
    let (a, b) = albedo_table().lookup(nov, roughness);
    let e = (a + b).clamp(1e-2, 1.0);
    let mut out = [0.0f32; 3];
    for c in 0..3 {
        let single = f0[c] * a + b;
        let compensation = 1.0 + f0[c] * (1.0 / e - 1.0);
        out[c] = (single * compensation).clamp(0.0, 1.0);
    }
    out
}

/// The BRDF, per steradian, for one light direction.
///
/// `n`, `v` and `l` are unit vectors in the same space: the surface normal, the
/// direction toward the viewer, and the direction toward the light. No cosine
/// term and no illuminance — see the module docs on why those belong to the
/// caller.
///
/// A light at or below the horizon returns zero rather than a negative number,
/// which is what stops a back-facing light subtracting light from a surface.
pub fn evaluate(surface: &Surface, n: glam::Vec3, v: glam::Vec3, l: glam::Vec3) -> Response {
    let nol = n.dot(l);
    let nov = n.dot(v);
    if nol <= 0.0 || nov <= 0.0 {
        return Response::ZERO;
    }
    // Clamped rather than trusted: an interpolated normal on a coarse mesh can
    // face slightly away from a viewer who can see the triangle, and the
    // grazing terms below divide by these.
    let nol = nol.min(1.0);
    let nov = nov.clamp(1e-4, 1.0);

    // Plain `normalize`, and the early return above is what makes it safe.
    //
    // `v + l` vanishes only when `l = -v`, and that forces `n·l = -(n·v)`, so
    // the two cannot both be positive — the return above has already taken
    // that case. A falsification sweep proved the guard that used to be here
    // was dead, on both this side and the shader's.
    //
    // Which means this line depends on that early return. Anyone relaxing it —
    // to shade a two-sided surface, say — brings the NaN back, and a NaN in a
    // fragment shader spreads through the blend and is not traceable to a
    // light.
    let h = (v + l).normalize();
    let noh = n.dot(h).clamp(0.0, 1.0);
    let voh = v.dot(h).clamp(0.0, 1.0);

    let alpha = surface.alpha();
    let f0 = surface.f0();
    let f = fresnel(f0, voh);
    let d = distribution(noh, alpha);
    let vis = visibility(nov, nol, alpha);
    let compensation = energy_compensation(f0, nov, surface.roughness.clamp(0.0, 1.0));

    let specular = [
        d * vis * f[0] * compensation[0],
        d * vis * f[1] * compensation[1],
        d * vis * f[2] * compensation[2],
    ];

    // Lambert, coupled to what the specular lobe took.
    //
    // Kelemen–Szirmay-Kalos: `(1 - E(NoV))(1 - E(NoL)) / (1 - E_avg)`. Two
    // properties, and neither of the obvious forms has both.
    //
    // - **Symmetric**, so the BRDF stays reciprocal. Scaling by `1 - E(NoV)`
    //   alone conserves energy and is not.
    // - **Conserving**, because the hemispherical integral of that expression
    //   is exactly `albedo·(1 - E(NoV))` — the energy the specular lobe left.
    //   Scaling by `1 - F(VoH)` is reciprocal, since the half-vector is
    //   symmetric, and gains a third of the arriving light at a grazing view.
    let roughness = surface.roughness.clamp(0.0, 1.0);
    let share_v = specular_share(f0, nov, roughness);
    let share_l = specular_share(f0, nol, roughness);
    let (avg_a, avg_b) = albedo_table().average(roughness);
    let albedo = surface.diffuse_albedo();
    let inv_pi = std::f32::consts::FRAC_1_PI;
    let mut diffuse = [0.0f32; 3];
    for c in 0..3 {
        // The average share for this material. Uncompensated, which is a
        // choice and an exact one where it matters: the coupling only does
        // anything when there is a diffuse lobe, and then the material is not
        // a metal and `F0` is four percent, where the compensation is one to
        // within a thousandth.
        let avg = (f0[c] * avg_a + avg_b).clamp(0.0, 0.99);
        diffuse[c] =
            albedo[c] * inv_pi * (1.0 - share_v[c]) * (1.0 - share_l[c]) / (1.0 - avg);
    }

    Response { diffuse, specular }
}

/// What a fragment adds for one light: the BRDF, the cosine, the illuminance.
///
/// `illuminance` is what `lights::GpuLight::illuminance_at` answers — lux
/// arriving perpendicular to the light's direction — so the `NoL` here is the
/// projection onto the surface and nothing else applies it.
pub fn shade(
    surface: &Surface,
    n: glam::Vec3,
    v: glam::Vec3,
    l: glam::Vec3,
    illuminance: [f32; 3],
    diffuse_on: bool,
    specular_on: bool,
) -> [f32; 3] {
    let nol = n.dot(l).max(0.0);
    if nol <= 0.0 {
        return [0.0; 3];
    }
    let response = evaluate(surface, n, v, l);
    let mut out = [0.0f32; 3];
    for c in 0..3 {
        let mut f = 0.0;
        if diffuse_on {
            f += response.diffuse[c];
        }
        if specular_on {
            f += response.specular[c];
        }
        out[c] = f * illuminance[c] * nol;
    }
    out
}

/// The shader source, for a pass that shades with this BRDF.
///
/// There is no `#import` in this build, so a shader needing `brdf_evaluate`
/// takes the text. A function exposed rather than the file copied, because a
/// copy is the drift the device test exists to prevent.
pub fn shader_source() -> &'static str {
    include_str!("shaders/brdf.wgsl")
}

/// Where this library expects the DFG table to be bound.
pub const DFG_GROUP: u32 = 2;
pub const DFG_BINDING: u32 = 0;

/// Where the probe pass begins.
///
/// Everything after it is for the device test and is stripped when this file is
/// composed into another shader: the probe declares bindings a forward pass
/// does not have, so leaving them in would make the pass demand buffers nobody
/// binds.
const PROBE_MARKER: &str = "// --- the probe pass";

/// The part of the shader another shader can include.
///
/// # Panics
///
/// If the marker is gone. Returning the whole file instead would compose the
/// probe's bindings into a forward pass, and the failure would arrive as a
/// pipeline validation error a long way from the edit that caused it.
pub fn library_source() -> &'static str {
    let source = shader_source();
    match source.split_once(PROBE_MARKER) {
        Some((library, _)) => library,
        None => panic!("the probe marker is gone from this shader"),
    }
}


/// How many `vec2` entries the shader's buffer holds.
///
/// The table, then its per-roughness averages. One buffer rather than two,
/// because the forward pass will already be carrying lights, materials and a
/// camera, and a binding saved there is a binding not spent.
pub const DFG_ENTRIES: usize = ALBEDO_LUT * ALBEDO_LUT + ALBEDO_LUT;

/// The buffer the shader reads: the table, then the averages.
///
/// Interleaved as `(a, b)` pairs, row-major in roughness, which is what
/// `dfg_lookup` indexes. Built here rather than in the pass so that the layout
/// has one author — a table and a shader that disagree about the order produce
/// a picture that is merely wrong.
pub fn dfg_buffer(table: &AlbedoTable) -> Vec<f32> {
    let mut out = table.as_flat();
    out.extend(table.averages_flat());
    debug_assert_eq!(out.len(), DFG_ENTRIES * 2);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The directional albedo: how much of the light arriving from every
    /// direction leaves again, for a viewer at `nov`.
    ///
    /// A uniform hemisphere of unit radiance, integrated numerically. This is
    /// the furnace test, and it is the only way to check an energy claim — no
    /// single evaluation of a BRDF says anything about conservation.
    fn directional_albedo(surface: &Surface, nov: f32, which: Lobe) -> f32 {
        let n = glam::Vec3::Z;
        let theta_v = nov.clamp(1e-3, 1.0).acos();
        let v = glam::Vec3::new(theta_v.sin(), 0.0, theta_v.cos());

        // A cosine-weighted sum over the hemisphere. The grid is fine enough
        // that a smooth lobe is resolved; the smoothest case is checked
        // separately, because a near-delta lobe cannot be integrated on a grid
        // at all and pretending otherwise would be a test that lies.
        let steps_theta = 256;
        let steps_phi = 512;
        let mut total = 0.0f64;
        for i in 0..steps_theta {
            let theta = (i as f32 + 0.5) / steps_theta as f32 * std::f32::consts::FRAC_PI_2;
            let (sin_t, cos_t) = theta.sin_cos();
            let d_theta = std::f32::consts::FRAC_PI_2 / steps_theta as f32;
            for j in 0..steps_phi {
                let phi = (j as f32 + 0.5) / steps_phi as f32 * std::f32::consts::TAU;
                let d_phi = std::f32::consts::TAU / steps_phi as f32;
                let l = glam::Vec3::new(sin_t * phi.cos(), sin_t * phi.sin(), cos_t);
                let r = evaluate(surface, n, v, l);
                let f = match which {
                    Lobe::Specular => r.specular[0],
                    Lobe::Diffuse => r.diffuse[0],
                    Lobe::Both => r.total()[0],
                };
                total += (f * cos_t * sin_t * d_theta * d_phi) as f64;
            }
        }
        total as f32
    }

    #[derive(Clone, Copy)]
    enum Lobe {
        Specular,
        Diffuse,
        Both,
    }

    fn white_metal(roughness: f32) -> Surface {
        Surface {
            base_color: SceneRgb::WHITE,
            metallic: 1.0,
            roughness,
        }
    }

    #[test]
    fn a_white_metal_keeps_its_energy_at_every_roughness() {
        // The test §2.3's requirement turns into. A microfacet model traces one
        // bounce, so what would have bounced again is lost, and the loss grows
        // with roughness — a rough metal comes out visibly darker than it
        // should. The compensation puts it back.
        //
        // This is also why the compensation could not be deferred: switching it
        // on later brightens every rough metal in every saved scene, and the
        // values an artist chose against the darker model become wrong.
        for roughness in [0.1, 0.25, 0.4, 0.6, 0.8, 1.0] {
            let albedo = directional_albedo(&white_metal(roughness), 0.7, Lobe::Specular);
            assert!(
                (0.93..=1.02).contains(&albedo),
                "roughness {roughness} reflected {albedo}"
            );
        }
    }

    #[test]
    fn the_compensation_is_doing_the_work_and_not_the_fit() {
        // Guards the test above: if the single-scattering lobe were already
        // conserving energy, that test would pass with the compensation
        // deleted and would be measuring nothing.
        //
        // So here is the same integral with the compensation removed. A rough
        // metal loses a fifth of its light, which is the darkening the plan
        // describes.
        let roughness = 0.8;
        let surface = white_metal(roughness);
        let with = directional_albedo(&surface, 0.7, Lobe::Specular);

        let compensated = energy_compensation(surface.f0(), 0.7, roughness);
        // The compensation is a scale on the lobe, so dividing it out gives
        // what the uncompensated model would have reflected.
        let without = with / compensated[0];
        assert!(
            without < 0.85,
            "single scattering lost only {}",
            1.0 - without
        );
        assert!(
            compensated[0] > 1.1,
            "the compensation barely scaled anything: {}",
            compensated[0]
        );
    }

    #[test]
    fn no_surface_reflects_more_than_arrives() {
        // The other half of conservation, and the easier one to violate: a
        // diffuse lobe that ignored what the specular took, or a compensation
        // that overshot, would show as a surface that glows.
        for metallic in [0.0, 0.5, 1.0] {
            for roughness in [0.05, 0.2, 0.5, 0.9, 1.0] {
                for nov in [0.15, 0.5, 0.95] {
                    let surface = Surface {
                        base_color: SceneRgb::WHITE,
                        metallic,
                        roughness,
                    };
                    let albedo = directional_albedo(&surface, nov, Lobe::Both);
                    assert!(
                        albedo <= 1.02,
                        "metallic {metallic} roughness {roughness} at {nov} reflected {albedo}"
                    );
                }
            }
        }
    }

    #[test]
    fn the_albedo_table_depends_on_the_view_angle() {
        // The mistake that started the table. Lazarov's analytic fit looks like
        // it supplies this, and its `scale + bias` is `1 − 0.55·roughness` — a
        // straight line with the view-dependent term cancelled exactly. The
        // truth varies by a factor of two across view angles at high
        // roughness, which is precisely what the fit throws away.
        let table = albedo_table();
        for roughness in [0.4, 0.7, 1.0] {
            let (a, b) = table.lookup(0.95, roughness);
            let head_on = a + b;
            let (a, b) = table.lookup(0.1, roughness);
            let grazing = a + b;
            assert!(
                (head_on - grazing).abs() > 0.05,
                "roughness {roughness} is flat in the view angle: {head_on} against {grazing}"
            );
        }
    }

    #[test]
    fn the_albedo_table_is_a_fraction_and_falls_with_roughness() {
        // It is a fraction of the arriving light, so it cannot exceed one — and
        // it must fall as the surface roughens, because that is the energy the
        // compensation exists to put back.
        let table = albedo_table();
        for nov in [0.2, 0.5, 0.9] {
            let mut previous = f32::INFINITY;
            for roughness in [0.05, 0.2, 0.4, 0.6, 0.8, 1.0] {
                let (a, b) = table.lookup(nov, roughness);
                let e = a + b;
                assert!((0.0..=1.0).contains(&e), "E({nov}, {roughness}) = {e}");
                assert!(e < previous + 1e-3, "E rose at roughness {roughness}");
                previous = e;
            }
            // And a nearly smooth surface loses almost nothing.
            let (a, b) = table.lookup(nov, MIN_ROUGHNESS);
            assert!(a + b > 0.9, "a mirror lost {}", 1.0 - (a + b));
        }
    }

    #[test]
    fn the_two_channels_reconstruct_the_fresnel_weighted_albedo() {
        // Which is what the second channel is *for*: `F0·a + b`. A single
        // channel gives only `a + b`, and the diffuse coupling needs to know
        // how much a *particular* material took.
        let table = albedo_table();
        let (a, b) = table.lookup(0.6, 0.3);
        // A mirror takes everything the lobe can deliver.
        assert!(((a + b) - (1.0 * a + b)).abs() < 1e-6);
        // A dielectric takes far less, and more than its four percent, because
        // Fresnel rises toward grazing within the lobe.
        let dielectric = DIELECTRIC_F0 * a + b;
        assert!(dielectric > DIELECTRIC_F0, "{dielectric}");
        assert!(dielectric < a + b, "{dielectric} against {}", a + b);
    }

    #[test]
    fn the_averages_are_the_rows_own_averages() {
        // Reduced from the 2D table rather than integrated again, so the two
        // cannot disagree — and the coupled diffuse divides by this, so a
        // disagreement would be an energy leak with no other symptom.
        let table = albedo_table();
        for roughness in [0.1, 0.5, 0.9] {
            let (avg_a, avg_b) = table.average(roughness);
            // A cosine-weighted average of values in [0, 1] is in [0, 1].
            assert!((0.0..=1.0).contains(&avg_a), "{avg_a}");
            assert!((0.0..=1.0).contains(&avg_b), "{avg_b}");
            // And it lies between the row's extremes.
            let (a_head, b_head) = table.lookup(0.98, roughness);
            let (a_graze, b_graze) = table.lookup(0.02, roughness);
            let lo = (a_head + b_head).min(a_graze + b_graze);
            let hi = (a_head + b_head).max(a_graze + b_graze);
            let avg = avg_a + avg_b;
            assert!(
                avg >= lo - 0.05 && avg <= hi + 0.05,
                "roughness {roughness}: average {avg} outside [{lo}, {hi}]"
            );
        }
    }

    #[test]
    fn the_diffuse_coupling_is_symmetric_in_the_two_directions() {
        // Which is what keeps the whole BRDF reciprocal. Scaling by
        // `1 - E(NoV)` alone conserves energy and is not symmetric, and the
        // reciprocity test caught exactly that.
        let surface = Surface {
            base_color: SceneRgb::new(0.8, 0.8, 0.8),
            metallic: 0.0,
            roughness: 0.4,
        };
        let n = glam::Vec3::Z;
        let a = glam::Vec3::new(0.9, 0.0, 0.436).normalize();
        let b = glam::Vec3::new(0.1, 0.2, 0.974).normalize();
        let forward = evaluate(&surface, n, a, b).diffuse[0];
        let backward = evaluate(&surface, n, b, a).diffuse[0];
        assert!(
            (forward - backward).abs() < forward * 1e-4,
            "{forward} against {backward}"
        );
    }

    #[test]
    fn the_table_is_the_same_on_every_run() {
        // Deterministic, because it is integrated with a Hammersley sequence
        // and not a random one. A table that differed between runs would make
        // every energy test above a flaky one.
        //
        // Weak on its own, and a sweep showed it: two computations in the same
        // process catch an actual random source and not much else. The
        // committed values below are what catch a table that differs between
        // *machines*, which is the thing the comment above was really claiming.
        let once = AlbedoTable::compute();
        let twice = AlbedoTable::compute();
        for roughness in [0.1, 0.37, 0.9] {
            for nov in [0.05, 0.4, 0.99] {
                assert_eq!(once.lookup(nov, roughness), twice.lookup(nov, roughness));
            }
        }
    }

    #[test]
    fn the_table_is_what_it_was_measured_to_be() {
        // A tripwire, in the sense the golden files are: a build whose table
        // differs makes a different picture, and nothing else says so.
        //
        // The tolerance is not arbitrary. It is wider than the Monte Carlo
        // error of five hundred samples and far narrower than anything that
        // changes the table — the analytic line this replaced was six percent
        // out, and every break in the sweep moved a value by more than a
        // percent.
        //
        // These numbers are also readable as physics, which is why they are
        // spelled out rather than hashed. A nearly smooth surface at a
        // head-on view reflects everything (`a` is one and `b` is nothing,
        // because Fresnel barely rises within so narrow a lobe). At a grazing
        // view the same surface has moved almost all of its weight into `b`,
        // which is the grazing reflection that makes a shallow view of water
        // mirror-like. And a rough surface has lost a third of its light,
        // which is what the compensation puts back.
        const TOLERANCE: f32 = 1e-3;
        let table = albedo_table();
        for (nov, roughness, a, b) in [
            (0.1f32, 0.1f32, 0.414_533, 0.578_614),
            (0.5, 0.1, 0.968_274, 0.031_726),
            (0.9, 0.1, 0.999_987, 0.000_013),
            (0.1, 0.5, 0.761_214, 0.140_154),
            (0.5, 0.5, 0.833_992, 0.022_562),
            (0.9, 0.5, 0.907_693, 0.000_353),
            (0.1, 0.95, 0.745_469, 0.027_652),
            (0.5, 0.95, 0.489_465, 0.003_689),
            (0.9, 0.95, 0.379_899, 0.000_200),
        ] {
            let (got_a, got_b) = table.lookup(nov, roughness);
            assert!(
                (got_a - a).abs() < TOLERANCE,
                "a at ({nov}, {roughness}): {got_a} against {a}"
            );
            assert!(
                (got_b - b).abs() < TOLERANCE,
                "b at ({nov}, {roughness}): {got_b} against {b}"
            );
        }

        for (roughness, a, b) in [
            (0.1f32, 0.952_497, 0.047_106),
            (0.5, 0.863_642, 0.018_785),
            (0.95, 0.451_463, 0.003_321),
        ] {
            let (got_a, got_b) = table.average(roughness);
            assert!(
                (got_a - a).abs() < TOLERANCE && (got_b - b).abs() < TOLERANCE,
                "average at {roughness}: ({got_a}, {got_b}) against ({a}, {b})"
            );
        }
    }

    #[test]
    fn the_shader_declares_the_same_constants() {
        // The four numbers both implementations have to agree on, and the kind
        // of disagreement a parity test would show as a puzzling offset rather
        // than as a mismatched constant.
        let source = shader_source();
        for declaration in [
            format!("const ALBEDO_LUT: u32 = {ALBEDO_LUT}u;"),
            format!("const DIELECTRIC_F0: f32 = {DIELECTRIC_F0};"),
            format!("const MIN_ROUGHNESS: f32 = {MIN_ROUGHNESS};"),
        ] {
            assert!(
                source.contains(&declaration),
                "the shader does not declare `{declaration}`"
            );
        }
    }

    #[test]
    fn the_shader_parses() {
        naga::front::wgsl::parse_str(shader_source()).expect("brdf.wgsl parses");
    }

    #[test]
    fn the_buffer_is_the_table_then_the_averages() {
        // The layout the shader indexes, asserted here so that the table and
        // the shader have one author rather than two conventions.
        let table = albedo_table();
        let buffer = dfg_buffer(table);
        assert_eq!(buffer.len(), DFG_ENTRIES * 2);

        // A cell of the table, at its documented offset.
        let (ri, vi) = (5usize, 11usize);
        let at = 2 * (ri * ALBEDO_LUT + vi);
        let nov = (vi as f32 + 0.5) / ALBEDO_LUT as f32;
        let roughness = (ri as f32 + 0.5) / ALBEDO_LUT as f32;
        let (a, b) = table.lookup(nov, roughness);
        assert!((buffer[at] - a).abs() < 1e-6);
        assert!((buffer[at + 1] - b).abs() < 1e-6);

        // And an average, after the table.
        let base = 2 * ALBEDO_LUT * ALBEDO_LUT;
        let (avg_a, avg_b) = table.average(roughness);
        assert!((buffer[base + 2 * ri] - avg_a).abs() < 1e-6);
        assert!((buffer[base + 2 * ri + 1] - avg_b).abs() < 1e-6);
    }

    #[test]
    fn the_flat_forms_are_what_a_shader_would_index() {
        // The table has to cross to the GPU, and the layout is the one place a
        // mirror can be wrong without any test of the arithmetic noticing.
        let table = albedo_table();
        let flat = table.as_flat();
        assert_eq!(flat.len(), ALBEDO_LUT * ALBEDO_LUT * 2);
        // Row-major in roughness, interleaved: cell (nov index, roughness
        // index) starts at `2·(roughness·LUT + nov)`.
        for (ri, roughness_index) in [0usize, 7, ALBEDO_LUT - 1].iter().enumerate() {
            let _ = ri;
            for nov_index in [0usize, 3, ALBEDO_LUT - 1] {
                let at = 2 * (roughness_index * ALBEDO_LUT + nov_index);
                let nov = (nov_index as f32 + 0.5) / ALBEDO_LUT as f32;
                let roughness = (*roughness_index as f32 + 0.5) / ALBEDO_LUT as f32;
                let (a, b) = table.lookup(nov, roughness);
                assert!((flat[at] - a).abs() < 1e-6, "a at {at}");
                assert!((flat[at + 1] - b).abs() < 1e-6, "b at {at}");
            }
        }

        let averages = table.averages_flat();
        assert_eq!(averages.len(), ALBEDO_LUT * 2);
    }

    #[test]
    fn a_metal_has_no_diffuse_lobe() {
        // Not an approximation: a conductor's electrons absorb what enters it,
        // so nothing scatters back out. A metal with a diffuse term is the
        // commonest way to make metal look like painted plastic.
        let surface = white_metal(0.4);
        assert_eq!(surface.diffuse_albedo(), [0.0; 3]);
        let r = evaluate(
            &surface,
            glam::Vec3::Z,
            glam::Vec3::new(0.3, 0.0, 0.954).normalize(),
            glam::Vec3::new(-0.3, 0.0, 0.954).normalize(),
        );
        assert_eq!(r.diffuse, [0.0; 3]);
        assert!(r.specular[0] > 0.0);
    }

    #[test]
    fn a_dielectric_reflects_four_percent_head_on_and_all_of_it_at_grazing() {
        // The two ends of Fresnel, and the reason a plastic sphere has a bright
        // rim. A model with a constant reflectance has no rim and looks like
        // clay.
        let head_on = fresnel([DIELECTRIC_F0; 3], 1.0);
        assert!((head_on[0] - DIELECTRIC_F0).abs() < 1e-6, "{head_on:?}");

        let grazing = fresnel([DIELECTRIC_F0; 3], 0.0);
        assert!((grazing[0] - 1.0).abs() < 1e-6, "{grazing:?}");

        // And it rises monotonically between, or the rim would have a band in
        // it.
        let mut previous = 0.0;
        for i in (0..=100).rev() {
            let f = fresnel([DIELECTRIC_F0; 3], i as f32 / 100.0)[0];
            assert!(f >= previous - 1e-6, "Fresnel turned back at {i}");
            previous = f;
        }
    }

    #[test]
    fn a_metals_colour_is_its_reflectance_and_a_dielectrics_is_not() {
        // The metal-roughness trick: one field means two things, and `metallic`
        // says which. Getting this backwards makes every metal grey and every
        // plastic mirror-coloured.
        let gold = Surface {
            base_color: SceneRgb::new(1.0, 0.77, 0.34),
            metallic: 1.0,
            roughness: 0.2,
        };
        assert_eq!(gold.f0(), [1.0, 0.77, 0.34]);
        assert_eq!(gold.diffuse_albedo(), [0.0; 3]);

        let plastic = Surface {
            base_color: SceneRgb::new(1.0, 0.77, 0.34),
            metallic: 0.0,
            roughness: 0.2,
        };
        assert_eq!(plastic.f0(), [DIELECTRIC_F0; 3]);
        assert_eq!(plastic.diffuse_albedo(), [1.0, 0.77, 0.34]);
    }

    #[test]
    fn the_brdf_is_reciprocal() {
        // Swapping the viewer and the light must not change the answer. It is a
        // property of every physical BRDF, it is what makes bidirectional path
        // tracing possible at all, and it is easy to lose in the visibility
        // term — the separable Smith form is *not* reciprocal.
        let surface = Surface {
            base_color: SceneRgb::new(0.9, 0.6, 0.3),
            metallic: 0.3,
            roughness: 0.35,
        };
        let n = glam::Vec3::Z;
        for (a, b) in [
            (
                glam::Vec3::new(0.4, 0.1, 0.91).normalize(),
                glam::Vec3::new(-0.6, 0.3, 0.74).normalize(),
            ),
            (
                glam::Vec3::new(0.9, 0.0, 0.44).normalize(),
                glam::Vec3::new(0.1, 0.7, 0.71).normalize(),
            ),
        ] {
            let forward = evaluate(&surface, n, a, b);
            let backward = evaluate(&surface, n, b, a);
            for c in 0..3 {
                let f = forward.total()[c];
                let r = backward.total()[c];
                assert!(
                    (f - r).abs() <= f.abs().max(r.abs()) * 1e-3,
                    "{f} against {r}"
                );
            }
        }
    }

    #[test]
    fn a_rougher_surface_has_a_wider_dimmer_highlight() {
        // Which is the whole meaning of the parameter. A peak that did not fall
        // as the lobe widened would be gaining energy; one that fell without
        // widening would be a surface getting darker for no reason.
        let n = glam::Vec3::Z;
        let v = glam::Vec3::Z;
        let mut previous_peak = f32::INFINITY;
        for roughness in [0.1, 0.2, 0.4, 0.7, 1.0] {
            let surface = white_metal(roughness);
            // The mirror direction, where the lobe peaks for this view.
            let peak = evaluate(&surface, n, v, glam::Vec3::Z).specular[0];
            assert!(peak < previous_peak, "roughness {roughness} peaked higher");
            previous_peak = peak;

            // And well off the mirror direction it is brighter than a smoother
            // surface would be there — the lobe got wider, not just dimmer.
            let off = glam::Vec3::new(0.5, 0.0, 0.866);
            let here = evaluate(&surface, n, v, off).specular[0];
            let smoother = evaluate(&white_metal(0.05), n, v, off).specular[0];
            assert!(here > smoother, "roughness {roughness} was not wider");
        }
    }

    #[test]
    fn a_light_below_the_horizon_contributes_nothing() {
        // Not a negative number. A back-facing light that subtracted would show
        // as a dark patch that moves with the light, which reads as a shadow
        // bug rather than as arithmetic.
        let surface = Surface::default();
        let n = glam::Vec3::Z;
        let v = glam::Vec3::Z;
        for l in [
            glam::Vec3::NEG_Z,
            glam::Vec3::new(0.7, 0.0, -0.71).normalize(),
            glam::Vec3::X,
        ] {
            let r = evaluate(&surface, n, v, l);
            assert_eq!(r, Response::ZERO, "{l} lit the back of a surface");
        }
    }

    #[test]
    fn a_viewer_behind_the_surface_sees_nothing() {
        // Reachable with an interpolated normal on a coarse mesh, and the
        // grazing terms divide by `NoV`.
        let r = evaluate(
            &Surface::default(),
            glam::Vec3::Z,
            glam::Vec3::NEG_Z,
            glam::Vec3::Z,
        );
        assert_eq!(r, Response::ZERO);
    }

    #[test]
    fn nothing_finite_produces_a_nan_or_an_infinity() {
        // The corners: a mirror, a fully rough surface, a light exactly on the
        // horizon, a viewer exactly edge-on, and parameters outside their
        // range because a file or a script can supply those.
        let awkward = [
            Surface {
                base_color: SceneRgb::WHITE,
                metallic: 1.0,
                roughness: 0.0,
            },
            Surface {
                base_color: SceneRgb::BLACK,
                metallic: 0.0,
                roughness: 1.0,
            },
            Surface {
                base_color: SceneRgb::new(2.0, -1.0, 0.5),
                metallic: 5.0,
                roughness: -3.0,
            },
            Surface {
                base_color: SceneRgb::new(f32::NAN, 0.5, 0.5),
                metallic: 0.5,
                roughness: 0.5,
            },
        ];
        let directions = [
            glam::Vec3::Z,
            glam::Vec3::new(1.0, 0.0, 1e-6).normalize(),
            glam::Vec3::new(0.0, 1.0, 1e-4).normalize(),
            glam::Vec3::new(0.577, 0.577, 0.577),
        ];
        for surface in awkward {
            for v in directions {
                for l in directions {
                    let r = evaluate(&surface, glam::Vec3::Z, v, l);
                    for c in 0..3 {
                        // A NaN base colour is allowed to produce a NaN
                        // response — it is not this function's business to
                        // launder a material — but an infinity is not, because
                        // that is the model dividing by something it should
                        // have clamped.
                        let nan_in = surface.base_color.to_array()[c].is_nan();
                        assert!(
                            nan_in
                                || (r.diffuse[c].is_finite() && r.specular[c].is_finite()),
                            "{surface:?} v {v} l {l} gave {r:?}"
                        );
                        assert!(
                            r.specular[c] != f32::INFINITY,
                            "{surface:?} v {v} l {l} gave an infinity"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn a_mirror_is_clamped_rather_than_infinite() {
        // A zero-roughness lobe is a delta function, and a punctual light
        // sampling one is either a miss or an unbounded value. The floor is
        // what makes the answer a number, and it has to be small enough that
        // the surface still reads as a mirror.
        let mirror = Surface {
            base_color: SceneRgb::WHITE,
            metallic: 1.0,
            roughness: 0.0,
        };
        assert_eq!(mirror.alpha(), MIN_ROUGHNESS * MIN_ROUGHNESS);
        let peak = evaluate(&mirror, glam::Vec3::Z, glam::Vec3::Z, glam::Vec3::Z).specular[0];
        assert!(peak.is_finite() && peak > 100.0, "{peak}");
        // Still sharp: an eighth of a degree off the mirror direction and it
        // has fallen away.
        let off = glam::Vec3::new(0.05, 0.0, 0.9987).normalize();
        let beside = evaluate(&mirror, glam::Vec3::Z, glam::Vec3::Z, off).specular[0];
        assert!(beside < peak * 0.5, "{beside} against {peak}");
    }

    #[test]
    fn a_lambertian_diffuse_integrates_to_its_albedo() {
        // The sanity check on the diffuse half: a white non-metal with a
        // smooth-enough specular lobe should send back nearly all of what
        // arrives, less what the specular reflected.
        let surface = Surface {
            base_color: SceneRgb::WHITE,
            metallic: 0.0,
            roughness: 0.5,
        };
        let diffuse = directional_albedo(&surface, 0.9, Lobe::Diffuse);
        // Less than one, because the specular lobe took its share first.
        assert!((0.85..0.99).contains(&diffuse), "{diffuse}");
        let both = directional_albedo(&surface, 0.9, Lobe::Both);
        assert!(both > diffuse, "the specular lobe contributed nothing");
        assert!(both <= 1.02, "{both}");
    }

    #[test]
    fn shading_applies_the_cosine_and_the_illuminance_once_each() {
        // The seam between this and `lights`: `illuminance_at` answers with lux
        // perpendicular to the light, so the projection onto the surface
        // happens here and nowhere else. Applied twice, every surface would be
        // darkest where it faces the light most directly.
        let surface = Surface::default();
        let n = glam::Vec3::Z;
        let v = glam::Vec3::Z;
        let l = glam::Vec3::new(0.6, 0.0, 0.8).normalize();
        let response = evaluate(&surface, n, v, l);
        let lux = [100.0, 50.0, 25.0];

        let shaded = shade(&surface, n, v, l, lux, true, true);
        for c in 0..3 {
            let want = response.total()[c] * lux[c] * n.dot(l);
            assert!((shaded[c] - want).abs() < want * 1e-5, "{shaded:?}");
        }
    }

    #[test]
    fn the_two_lobes_can_be_switched_independently() {
        // §2.2's lighting cheats reach this far: a rim light that must not blow
        // out a highlight sets `affect_specular` false, and that has to arrive
        // as a lobe not being added.
        let surface = Surface {
            base_color: SceneRgb::new(0.8, 0.8, 0.8),
            metallic: 0.0,
            roughness: 0.3,
        };
        let n = glam::Vec3::Z;
        let v = glam::Vec3::Z;
        let l = glam::Vec3::new(0.3, 0.0, 0.954).normalize();
        let lux = [100.0; 3];

        let both = shade(&surface, n, v, l, lux, true, true)[0];
        let only_diffuse = shade(&surface, n, v, l, lux, true, false)[0];
        let only_specular = shade(&surface, n, v, l, lux, false, true)[0];
        let neither = shade(&surface, n, v, l, lux, false, false)[0];

        assert_eq!(neither, 0.0);
        assert!(only_diffuse > 0.0 && only_specular > 0.0);
        assert!((only_diffuse + only_specular - both).abs() < both * 1e-5);
    }
}
