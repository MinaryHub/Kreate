//! Help ▸ Report an Issue and Help ▸ Ask a Question: written in Kreate,
//! posted on GitHub.
//!
//! The form in Kreate gathers what the person writes and what a developer
//! will want with it — the version, the platform, the graphics card — and
//! opens GitHub in the browser with all of it filled in: a new issue for a
//! problem, a new discussion for a question or an idea. The person posts it
//! there, signed in as themselves, and the answer comes back to them there.
//!
//! Nothing is sent from Kreate itself: no key, no relay, no address held in
//! the program. The link is the whole of it.

/// The repository the links go to. Set `KREATE_FEEDBACK_REPO` (`owner/name`)
/// when building to point a build at another.
pub const REPO: &str = match option_env!("KREATE_FEEDBACK_REPO") {
    Some(repo) => repo,
    None => "MinaryHub/Kreate",
};

/// What is being written, in the order the form offers them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Something is wrong: an issue, on the bug report form.
    Issue,
    /// A question: a discussion in Q&A.
    Question,
    /// An idea, a feature wished for: a discussion in Ideas.
    Idea,
}

impl Kind {
    pub const ALL: [Kind; 3] = [Kind::Issue, Kind::Question, Kind::Idea];

    pub fn from_index(i: usize) -> Kind {
        Self::ALL[i.min(Self::ALL.len() - 1)]
    }
}

/// What the form has.
#[derive(Clone, Debug, PartialEq)]
pub struct Feedback {
    pub kind: Kind,
    pub title: String,
    pub message: String,
    /// The version, platform and graphics card, or `None` to leave them out.
    pub details: Option<Details>,
}

/// What a developer wants to know about the Kreate it came from.
#[derive(Clone, Debug, PartialEq)]
pub struct Details {
    pub version: String,
    pub platform: String,
    pub gpu: String,
}

/// The most a message takes of a link, encoded: past about 8 000
/// characters a browser or GitHub turns a link away, and Hangul is nine
/// characters a syllable once percent-encoded. The rest is said to be
/// pasted on the page.
const MOST_MESSAGE: usize = 5000;

/// The page that posts `f`, its fields filled in.
pub fn url(f: &Feedback) -> String {
    let (message, cut) = shorten(f.message.trim());
    let title = f.title.trim();
    match f.kind {
        // The bug report form (.github/ISSUE_TEMPLATE/bug_report.yml) takes
        // its fields by their ids.
        Kind::Issue => {
            let mut q = vec![
                ("template", "bug_report.yml".to_string()),
                ("title", title.to_string()),
                ("what-happened", message),
            ];
            if let Some(d) = &f.details {
                q.push(("version", d.version.clone()));
                q.push(("platform", format!("{}\n{}", d.platform, d.gpu)));
            }
            if cut {
                q.push((
                    "steps",
                    "(The message was cut short in the link: paste the rest here.)".into(),
                ));
            }
            format!("https://github.com/{REPO}/issues/new?{}", query(&q))
        }
        Kind::Question | Kind::Idea => {
            let category = if f.kind == Kind::Question {
                "q-a"
            } else {
                "ideas"
            };
            let mut body = message;
            if cut {
                body.push_str("\n\n(The message was cut short in the link: paste the rest here.)");
            }
            if let Some(d) = &f.details {
                body.push_str(&format!(
                    "\n\n---\nKreate {} · {} · {}",
                    d.version, d.platform, d.gpu
                ));
            }
            format!(
                "https://github.com/{REPO}/discussions/new?{}",
                query(&[
                    ("category", category.to_string()),
                    ("title", title.to_string()),
                    ("body", body),
                ])
            )
        }
    }
}

/// As much of the message as encodes to at most `MOST_MESSAGE`, on a
/// character boundary, and whether it was cut.
fn shorten(message: &str) -> (String, bool) {
    let mut used = 0;
    let mut kept = String::new();
    for c in message.chars() {
        used += encode(c.encode_utf8(&mut [0; 4])).len();
        if used > MOST_MESSAGE {
            return (kept, true);
        }
        kept.push(c);
    }
    (kept, false)
}

fn query(pairs: &[(&str, String)]) -> String {
    pairs
        .iter()
        .filter(|(_, v)| !v.is_empty())
        .map(|(k, v)| format!("{k}={}", encode(v)))
        .collect::<Vec<_>>()
        .join("&")
}

/// Percent-encoding for a query value: everything but the unreserved
/// characters, byte by byte in UTF-8.
pub fn encode(text: &str) -> String {
    let mut out = String::with_capacity(text.len() * 3);
    for b in text.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn details() -> Option<Details> {
        Some(Details {
            version: "0.9.0".into(),
            platform: "linux-x86_64".into(),
            gpu: "llvmpipe".into(),
        })
    }

    #[test]
    fn an_issue_opens_the_bug_form_filled_in() {
        let u = url(&Feedback {
            kind: Kind::Issue,
            title: "Fillet fails".into(),
            message: "모서리 & 구멍".into(),
            details: details(),
        });
        assert!(u.starts_with(&format!(
            "https://github.com/{REPO}/issues/new?template=bug_report.yml&title=Fillet%20fails"
        )));
        assert!(
            u.contains("what-happened=%EB%AA%A8%EC%84%9C%EB%A6%AC%20%26%20"),
            "{u}"
        );
        assert!(u.contains("version=0.9.0"));
        assert!(u.contains("platform=linux-x86_64%0Allvmpipe"));
        // No raw separators from the message: one parameter cannot spill
        // into the next.
        assert!(!u.contains("& "));
    }

    #[test]
    fn a_question_and_an_idea_open_discussions_in_their_categories() {
        let q = url(&Feedback {
            kind: Kind::Question,
            title: "How do I mirror?".into(),
            message: "?".into(),
            details: None,
        });
        assert!(q.starts_with(&format!(
            "https://github.com/{REPO}/discussions/new?category=q-a&title=How%20do%20I%20mirror%3F"
        )));
        assert!(!q.contains("Kreate%200.9"));
        let i = url(&Feedback {
            kind: Kind::Idea,
            title: "Threads".into(),
            message: "please".into(),
            details: details(),
        });
        assert!(i.contains("category=ideas"));
        assert!(i.contains("Kreate%200.9.0"), "{i}");
    }

    #[test]
    fn a_long_message_is_cut_and_says_so() {
        let u = url(&Feedback {
            kind: Kind::Question,
            title: "t".into(),
            message: "가".repeat(5000),
            details: None,
        });
        assert!(u.len() < 8000, "{}", u.len());
        assert!(u.contains("cut%20short"));
    }
}
