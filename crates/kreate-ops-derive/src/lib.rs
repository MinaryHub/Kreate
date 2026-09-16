//! `#[derive(OpParams)]`.
//!
//! Turns an ordinary struct into a reflected parameter set: a static schema plus
//! conversion to and from a `PropertyMap`. An operator author writes
//!
//! ```ignore
//! #[derive(Clone, Default, OpParams)]
//! pub struct BevelParams {
//!     /// Width of the bevel.
//!     #[param(min = 0.0, soft_max = 1.0, unit = "length")]
//!     pub width: f64,
//!     /// Number of segments across the bevel.
//!     #[param(min = 1, max = 64, unit = "count")]
//!     pub segments: i64,
//! }
//! ```
//!
//! and gets the redo panel, the property editor, the scripting binding, the
//! keymap payload, the undo record, the i18n keys, and the golden-test harness
//! from it. Doc comments become the fallback tooltips, so documentation and UI
//! text cannot drift apart.
//!
//! Written without `syn` or `quote`. Those are excellent crates, but this macro
//! runs on every build of every crate that defines an operator, and there will
//! eventually be hundreds; the parsing needed here is small and specific, and
//! compile time in a mesh kernel is a real productivity tax.

use proc_macro::{Delimiter, Group, TokenStream, TokenTree};

/// Derives the reflected parameter schema for a struct.
#[proc_macro_derive(OpParams, attributes(param))]
pub fn derive_op_params(input: TokenStream) -> TokenStream {
    match expand(input) {
        Ok(ts) => ts,
        Err(msg) => format!("compile_error!({:?});", msg).parse().unwrap(),
    }
}

#[derive(Default, Clone)]
struct FieldAttrs {
    min: Option<String>,
    max: Option<String>,
    soft_min: Option<String>,
    soft_max: Option<String>,
    step: Option<String>,
    unit: Option<String>,
    /// For enum-typed fields: the variant names shown in the UI.
    variants: Option<Vec<String>>,
    /// Whether a three-component field is a colour rather than a direction.
    ///
    /// Only the *kind* changes; the value stays three numbers, so scripting,
    /// keymaps, undo and every file already written see exactly what they saw
    /// before. What changes is that the editor stops labelling the components
    /// X, Y and Z — a base colour is not a direction, and calling red "X" is
    /// the interface asserting something untrue about the value.
    color: bool,
}

struct Field {
    name: String,
    ty: String,
    doc: String,
    attrs: FieldAttrs,
}

fn expand(input: TokenStream) -> Result<TokenStream, String> {
    let tokens: Vec<TokenTree> = input.into_iter().collect();
    let struct_name = find_struct_name(&tokens)?;
    let body = find_struct_body(&tokens)?;
    let fields = parse_fields(&body)?;

    if fields.is_empty() {
        // A parameterless operator is legitimate (`mesh.select_all` with no
        // arguments is one), so this is an empty schema rather than an error.
        return Ok(emit(&struct_name, &fields));
    }
    Ok(emit(&struct_name, &fields))
}

fn find_struct_name(tokens: &[TokenTree]) -> Result<String, String> {
    let mut it = tokens.iter().peekable();
    while let Some(t) = it.next() {
        if let TokenTree::Ident(id) = t {
            if id.to_string() == "struct" {
                return match it.next() {
                    Some(TokenTree::Ident(name)) => Ok(name.to_string()),
                    _ => Err("expected a name after `struct`".into()),
                };
            }
        }
    }
    Err("OpParams can only be derived for a struct".into())
}

fn find_struct_body(tokens: &[TokenTree]) -> Result<Vec<TokenTree>, String> {
    for t in tokens {
        if let TokenTree::Group(g) = t {
            if g.delimiter() == Delimiter::Brace {
                return Ok(g.stream().into_iter().collect());
            }
        }
    }
    Err("OpParams requires a struct with named fields".into())
}

/// Splits the struct body into fields, collecting doc comments and `#[param]`.
fn parse_fields(body: &[TokenTree]) -> Result<Vec<Field>, String> {
    let mut fields = Vec::new();
    let mut docs: Vec<String> = Vec::new();
    let mut attrs = FieldAttrs::default();
    let mut i = 0;

    while i < body.len() {
        // Attributes: `#` followed by a bracketed group.
        if matches!(&body[i], TokenTree::Punct(p) if p.as_char() == '#') {
            let Some(TokenTree::Group(g)) = body.get(i + 1) else {
                return Err("malformed attribute".into());
            };
            let inner: Vec<TokenTree> = g.stream().into_iter().collect();
            match inner.first() {
                Some(TokenTree::Ident(id)) if id.to_string() == "doc" => {
                    if let Some(text) = inner.iter().find_map(literal_string) {
                        docs.push(text.trim().to_string());
                    }
                }
                Some(TokenTree::Ident(id)) if id.to_string() == "param" => {
                    if let Some(TokenTree::Group(args)) = inner.get(1) {
                        parse_param_attr(args, &mut attrs)?;
                    }
                }
                _ => {}
            }
            i += 2;
            continue;
        }

        // Visibility, then `name : Type ,`.
        if matches!(&body[i], TokenTree::Ident(id) if id.to_string() == "pub") {
            i += 1;
            // `pub(crate)` and friends.
            if matches!(body.get(i), Some(TokenTree::Group(g)) if g.delimiter() == Delimiter::Parenthesis)
            {
                i += 1;
            }
            continue;
        }

        let TokenTree::Ident(name) = &body[i] else {
            i += 1;
            continue;
        };
        if !matches!(body.get(i + 1), Some(TokenTree::Punct(p)) if p.as_char() == ':') {
            i += 1;
            continue;
        }

        // Everything up to the next top-level comma is the type.
        let mut ty = String::new();
        let mut j = i + 2;
        let mut depth = 0usize;
        while j < body.len() {
            match &body[j] {
                TokenTree::Punct(p) if p.as_char() == ',' && depth == 0 => break,
                TokenTree::Punct(p) if p.as_char() == '<' => {
                    depth += 1;
                    ty.push('<');
                }
                TokenTree::Punct(p) if p.as_char() == '>' => {
                    depth = depth.saturating_sub(1);
                    ty.push('>');
                }
                t => ty.push_str(&t.to_string()),
            }
            j += 1;
        }

        fields.push(Field {
            name: name.to_string(),
            ty: ty.replace(' ', ""),
            doc: docs.join(" "),
            attrs: std::mem::take(&mut attrs),
        });
        docs.clear();
        i = j + 1;
    }

    Ok(fields)
}

fn literal_string(t: &TokenTree) -> Option<String> {
    match t {
        TokenTree::Literal(l) => {
            let s = l.to_string();
            let s = s.strip_prefix('"')?.strip_suffix('"')?;
            Some(s.to_string())
        }
        _ => None,
    }
}

/// Parses `#[param(min = 0.0, unit = "length", variants("a", "b"))]`.
fn parse_param_attr(args: &Group, out: &mut FieldAttrs) -> Result<(), String> {
    let tokens: Vec<TokenTree> = args.stream().into_iter().collect();
    let mut i = 0;
    while i < tokens.len() {
        let TokenTree::Ident(key) = &tokens[i] else {
            i += 1;
            continue;
        };
        let key = key.to_string();

        // `variants("a", "b")`
        if key == "variants" {
            if let Some(TokenTree::Group(g)) = tokens.get(i + 1) {
                let list: Vec<String> = g
                    .stream()
                    .into_iter()
                    .filter_map(|t| literal_string(&t))
                    .collect();
                if list.is_empty() {
                    return Err("`variants(...)` needs at least one name".into());
                }
                out.variants = Some(list);
                i += 2;
                continue;
            }
            return Err("`variants` must be followed by a parenthesised list".into());
        }

        // `color`, a flag with no value
        if key == "color" {
            out.color = true;
            i += 1;
            continue;
        }

        // `key = value`
        if !matches!(tokens.get(i + 1), Some(TokenTree::Punct(p)) if p.as_char() == '=') {
            i += 1;
            continue;
        }
        let mut value = String::new();
        let mut j = i + 2;
        // A negative literal arrives as two tokens.
        while j < tokens.len() {
            match &tokens[j] {
                TokenTree::Punct(p) if p.as_char() == ',' => break,
                t => value.push_str(&t.to_string()),
            }
            j += 1;
        }
        let value = value.trim().to_string();

        match key.as_str() {
            "min" => out.min = Some(value),
            "max" => out.max = Some(value),
            "soft_min" => out.soft_min = Some(value),
            "soft_max" => out.soft_max = Some(value),
            "step" => out.step = Some(value),
            "unit" => out.unit = Some(value.trim_matches('"').to_string()),
            other => {
                return Err(format!(
                    "unknown #[param] key `{other}`; expected one of min, max, \
                     soft_min, soft_max, step, unit, variants, color"
                ))
            }
        }
        i = j + 1;
    }
    Ok(())
}

/// Whether three numbers are a colour or a direction.
fn three_kind(field: &Field) -> String {
    if field.attrs.color {
        "PropertyKind::Color".into()
    } else {
        "PropertyKind::Vec3".into()
    }
}

/// Maps a Rust type to the reflected kind and the conversion code.
struct TypeInfo {
    kind: String,
    /// Expression building a `PropertyValue` from `self.<field>`.
    to_value: &'static str,
    /// Expression reading a field back out of a `PropertyValue`.
    from_value: &'static str,
}

fn type_info(field: &Field) -> Result<TypeInfo, String> {
    if let Some(variants) = &field.attrs.variants {
        let list = variants
            .iter()
            .map(|v| format!("{v:?}"))
            .collect::<Vec<_>>()
            .join(", ");
        return Ok(TypeInfo {
            kind: format!("PropertyKind::Enum {{ variants: &[{list}] }}"),
            to_value: "PropertyValue::Enum(self.FIELD as u32)",
            from_value: "v.as_int().map(|i| i as _)",
        });
    }
    let t = match field.ty.as_str() {
        "bool" => TypeInfo {
            kind: "PropertyKind::Bool".into(),
            to_value: "PropertyValue::Bool(self.FIELD)",
            from_value: "v.as_bool()",
        },
        "i8" | "i16" | "i32" | "i64" | "u8" | "u16" | "u32" | "usize" => TypeInfo {
            kind: "PropertyKind::Int".into(),
            to_value: "PropertyValue::Int(self.FIELD as i64)",
            from_value: "v.as_int().map(|i| i as _)",
        },
        "f32" | "f64" => TypeInfo {
            kind: "PropertyKind::Float".into(),
            to_value: "PropertyValue::Float(self.FIELD as f64)",
            from_value: "v.as_float().map(|f| f as _)",
        },
        "String" => TypeInfo {
            kind: "PropertyKind::Text".into(),
            to_value: "PropertyValue::Text(self.FIELD.clone())",
            from_value: "v.as_text().map(|s| s.to_string())",
        },
        "[f64;3]" => TypeInfo {
            kind: three_kind(field),
            to_value: "PropertyValue::Vec3(self.FIELD)",
            from_value: "v.as_vec3()",
        },
        "[f32;3]" => TypeInfo {
            kind: three_kind(field),
            to_value: "PropertyValue::Vec3([self.FIELD[0] as f64, self.FIELD[1] as f64, self.FIELD[2] as f64])",
            from_value: "v.as_vec3().map(|a| [a[0] as f32, a[1] as f32, a[2] as f32])",
        },
        "Vec<u32>" => TypeInfo {
            kind: "PropertyKind::Ints".into(),
            to_value: "PropertyValue::Ints(self.FIELD.clone())",
            from_value: "v.as_ints().map(|s| s.to_vec())",
        },
        other => {
            return Err(format!(
                "`{}: {other}` is not a supported parameter type. Parameters must be \
                 serialisable so that scripting, keymaps and undo all see the same \
                 values; supported types are bool, the integer types, f32/f64, \
                 String, [f32; 3], [f64; 3], Vec<u32>, or any integer field \
                 annotated with #[param(variants(...))].",
                field.name
            ))
        }
    };
    Ok(t)
}

fn opt(value: &Option<String>) -> String {
    match value {
        Some(v) => format!("Some({v} as f64)"),
        None => "None".into(),
    }
}

fn unit_expr(field: &Field) -> Result<String, String> {
    let name = field.attrs.unit.as_deref().unwrap_or("none");
    Ok(match name {
        "none" => "Unit::None",
        "length" => "Unit::Length",
        "angle" => "Unit::Angle",
        "factor" => "Unit::Factor",
        "count" => "Unit::Count",
        other => {
            return Err(format!(
                "unknown unit {other:?} on `{}`; expected none, length, angle, factor or count",
                field.name
            ))
        }
    }
    .to_string())
}

fn emit(struct_name: &str, fields: &[Field]) -> TokenStream {
    let mut schema = String::new();
    let mut to_map = String::new();
    let mut from_map = String::new();

    for f in fields {
        let info = match type_info(f) {
            Ok(i) => i,
            Err(e) => return format!("compile_error!({e:?});").parse().unwrap(),
        };
        let unit = match unit_expr(f) {
            Ok(u) => u,
            Err(e) => return format!("compile_error!({e:?});").parse().unwrap(),
        };

        schema.push_str(&format!(
            "PropertyDesc {{ name: {:?}, kind: {}, min: {}, max: {}, soft_min: {}, \
             soft_max: {}, step: {}, unit: {}, doc: {:?} }},",
            f.name,
            info.kind,
            opt(&f.attrs.min),
            opt(&f.attrs.max),
            opt(&f.attrs.soft_min),
            opt(&f.attrs.soft_max),
            opt(&f.attrs.step),
            unit,
            f.doc,
        ));

        to_map.push_str(&format!(
            "map.set({:?}, {});",
            f.name,
            info.to_value.replace("FIELD", &f.name)
        ));

        // Missing entries fall back to the default: a keymap or file written
        // against an older version is missing parameters added since, and must
        // still load rather than failing.
        from_map.push_str(&format!(
            "if let Some(v) = map.get({:?}) {{
                 let v = Self::schema()[{}].clamp(v.clone());
                 if let Some(x) = {} {{ out.{} = x; }}
             }}",
            f.name,
            schema_index(fields, &f.name),
            info.from_value,
            f.name,
        ));
    }

    let code = format!(
        r#"
impl ::kreate_ops::property::OpParams for {struct_name} {{
    fn schema() -> &'static [::kreate_ops::property::PropertyDesc] {{
        use ::kreate_ops::property::{{PropertyDesc, PropertyKind, Unit}};
        &[{schema}]
    }}

    fn to_map(&self) -> ::kreate_ops::property::PropertyMap {{
        use ::kreate_ops::property::PropertyValue;
        let mut map = ::kreate_ops::property::PropertyMap::new();
        {to_map}
        map
    }}

    fn from_map(map: &::kreate_ops::property::PropertyMap) -> Self {{
        #[allow(unused_imports)]
        use ::kreate_ops::property::PropertyValue;
        let mut out = Self::default();
        {from_map}
        out
    }}
}}
"#
    );
    code.parse().expect("generated code should parse")
}

fn schema_index(fields: &[Field], name: &str) -> usize {
    fields.iter().position(|f| f.name == name).unwrap_or(0)
}
