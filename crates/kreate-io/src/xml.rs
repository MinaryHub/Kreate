//! Just enough XML for the formats that are written in it — URDF and
//! COLLADA: elements, their attributes and their text, as a tree.
//!
//! Not a general parser, and not trying to be one: no DTDs, no namespaces
//! beyond keeping the prefix in the name, no entity but the five predefined
//! and numeric ones. What it does not understand it skips rather than
//! refuses — a comment, a processing instruction, a CDATA section's markup —
//! because the files it reads are written by exporters, not by hand, and an
//! exporter's flourish is no reason to lose a robot.
//!
//! Bounded as every reader here is: the depth of the tree is capped, so a
//! file of nothing but opening tags cannot exhaust the stack of whatever walks
//! it.

/// How deep elements may nest.
const MAX_DEPTH: usize = 256;

/// One element: its name, its attributes in order, its text (all of it,
/// concatenated) and its children.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Element {
    pub name: String,
    pub attributes: Vec<(String, String)>,
    pub text: String,
    pub children: Vec<Element>,
}

impl Element {
    /// The attribute called `name`.
    pub fn attr(&self, name: &str) -> Option<&str> {
        self.attributes
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    }

    /// The first child called `name`.
    pub fn child(&self, name: &str) -> Option<&Element> {
        self.children.iter().find(|c| c.name == name)
    }

    /// Every child called `name`.
    pub fn all<'a>(&'a self, name: &'a str) -> impl Iterator<Item = &'a Element> + 'a {
        self.children.iter().filter(move |c| c.name == name)
    }

    /// Every element below this one, depth first, this one included.
    pub fn descendants(&self) -> Vec<&Element> {
        let mut out = vec![self];
        let mut at = 0;
        while at < out.len() {
            let e = out[at];
            out.extend(e.children.iter());
            at += 1;
        }
        out
    }
}

/// Why a document could not be read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct XmlError(pub String);

impl std::fmt::Display for XmlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "XML: {}", self.0)
    }
}

impl std::error::Error for XmlError {}

/// The document's root element.
pub fn parse(text: &str) -> Result<Element, XmlError> {
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    let bytes = text.as_bytes();
    let mut stack: Vec<Element> = vec![Element::default()];
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != b'<' {
            let end = text[i..].find('<').map_or(bytes.len(), |n| i + n);
            let top = stack.last_mut().expect("the document");
            top.text.push_str(&unescape(&text[i..end]));
            i = end;
            continue;
        }
        let rest = &text[i..];
        if let Some(body) = rest.strip_prefix("<!--") {
            i += 4 + body.find("-->").map_or(body.len(), |n| n + 3);
            continue;
        }
        if let Some(body) = rest.strip_prefix("<![CDATA[") {
            let end = body.find("]]>").unwrap_or(body.len());
            stack
                .last_mut()
                .expect("the document")
                .text
                .push_str(&body[..end]);
            i += 9 + (end + 3).min(body.len());
            continue;
        }
        if rest.starts_with("<?") || rest.starts_with("<!") {
            i += rest.find('>').map_or(rest.len(), |n| n + 1);
            continue;
        }
        let close = tag_end(rest).ok_or_else(|| XmlError("a tag that never closes".into()))?;
        let tag = &rest[1..close];
        i += close + 1;
        if let Some(name) = tag.strip_prefix('/') {
            let name = name.trim();
            if stack.len() < 2 {
                return Err(XmlError(format!("`</{name}>` closes nothing")));
            }
            let done = stack.pop().expect("checked");
            if done.name != name {
                return Err(XmlError(format!("`</{name}>` closes `<{}>`", done.name)));
            }
            stack.last_mut().expect("checked").children.push(done);
            continue;
        }
        let (tag, empty) = match tag.strip_suffix('/') {
            Some(t) => (t, true),
            None => (tag, false),
        };
        let element = open(tag)?;
        if empty {
            stack
                .last_mut()
                .expect("the document")
                .children
                .push(element);
        } else {
            if stack.len() > MAX_DEPTH {
                return Err(XmlError("elements nested too deep".into()));
            }
            stack.push(element);
        }
    }
    if stack.len() != 1 {
        return Err(XmlError(format!(
            "`<{}>` is never closed",
            stack.last().map(|e| e.name.as_str()).unwrap_or("")
        )));
    }
    stack
        .pop()
        .and_then(|d| d.children.into_iter().next())
        .ok_or_else(|| XmlError("no root element".into()))
}

/// Where a tag's `>` is, past any quoted `>` inside its attributes.
fn tag_end(rest: &str) -> Option<usize> {
    let mut quote = None;
    for (n, c) in rest.char_indices().skip(1) {
        match (quote, c) {
            (None, '"' | '\'') => quote = Some(c),
            (Some(q), c) if c == q => quote = None,
            (None, '>') => return Some(n),
            _ => {}
        }
    }
    None
}

/// An opening tag's name and attributes.
fn open(tag: &str) -> Result<Element, XmlError> {
    let tag = tag.trim();
    let name_end = tag.find(|c: char| c.is_whitespace()).unwrap_or(tag.len());
    let mut element = Element {
        name: tag[..name_end].to_string(),
        ..Element::default()
    };
    if element.name.is_empty() {
        return Err(XmlError("a tag with no name".into()));
    }
    let mut rest = tag[name_end..].trim_start();
    while !rest.is_empty() {
        let eq = rest.find('=').ok_or_else(|| {
            XmlError(format!(
                "an attribute with no value in `<{}>`",
                element.name
            ))
        })?;
        let key = rest[..eq].trim().to_string();
        let after = rest[eq + 1..].trim_start();
        let quote = after
            .chars()
            .next()
            .filter(|c| *c == '"' || *c == '\'')
            .ok_or_else(|| XmlError(format!("`{key}`'s value is not quoted")))?;
        let body = &after[1..];
        let end = body
            .find(quote)
            .ok_or_else(|| XmlError(format!("`{key}`'s value never ends")))?;
        element.attributes.push((key, unescape(&body[..end])));
        rest = body[end + 1..].trim_start();
    }
    Ok(element)
}

/// The five predefined entities and numeric ones; anything else as it was.
fn unescape(text: &str) -> String {
    if !text.contains('&') {
        return text.to_string();
    }
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find('&') {
        out.push_str(&rest[..at]);
        rest = &rest[at..];
        let Some(end) = rest.find(';').filter(|e| *e <= 12) else {
            out.push('&');
            rest = &rest[1..];
            continue;
        };
        let entity = &rest[1..end];
        let decoded = match entity {
            "lt" => Some('<'),
            "gt" => Some('>'),
            "amp" => Some('&'),
            "quot" => Some('"'),
            "apos" => Some('\''),
            _ => entity
                .strip_prefix("#x")
                .and_then(|h| u32::from_str_radix(h, 16).ok())
                .or_else(|| entity.strip_prefix('#').and_then(|d| d.parse().ok()))
                .and_then(char::from_u32),
        };
        match decoded {
            Some(c) => {
                out.push(c);
                rest = &rest[end + 1..];
            }
            None => {
                out.push('&');
                rest = &rest[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

/// Numbers separated by white space, as XML formats write arrays; what does
/// not read is skipped.
pub fn numbers(text: &str) -> Vec<f64> {
    text.split_ascii_whitespace()
        .filter_map(|t| t.parse().ok())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tree_reads_with_attributes_text_and_entities() {
        let doc = parse(
            r#"<?xml version="1.0"?>
            <!-- a robot -->
            <robot name="r&amp;d">
              <link name='a > b'/>
              <float_array count="3">1 2.5 -3</float_array>
              <note><![CDATA[<raw>]]> &#65;&#x42;</note>
            </robot>"#,
        )
        .unwrap();
        assert_eq!(doc.name, "robot");
        assert_eq!(doc.attr("name"), Some("r&d"));
        assert_eq!(doc.child("link").unwrap().attr("name"), Some("a > b"));
        assert_eq!(
            numbers(&doc.child("float_array").unwrap().text),
            vec![1.0, 2.5, -3.0]
        );
        assert_eq!(doc.child("note").unwrap().text.trim(), "<raw> AB");
        assert_eq!(doc.descendants().len(), 4);
    }

    #[test]
    fn broken_documents_are_said_not_guessed() {
        assert!(parse("<a><b></a>").is_err());
        assert!(parse("<a>").is_err());
        assert!(parse("<a x=1/>").is_err());
        let deep = "<a>".repeat(1000);
        assert!(parse(&deep).is_err());
    }
}
