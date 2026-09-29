//! F3: find a command by typing its name.
//!
//! Blender's menu search. Every entry of every menu the current mode offers,
//! found by any part of its name, its menu path or its operator id, with its
//! shortcut beside it — which is how a shortcut gets learnt — and run with
//! Enter. The palette does not know what the entries do; the application hands
//! it a list and runs whichever one comes back, through the same path the menu
//! entry would have taken.
//!
//! # Matching
//!
//! Every word typed has to appear, in order of its letters, somewhere in the
//! entry. A word found whole beats one found scattered, and one at the start
//! of a word beats one in the middle, so "ext" finds Extrude before it finds
//! Next. The operator id is searched too, so a Korean interface still answers
//! to the English name a tutorial uses.

use egui::Key;

/// One searchable command.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PaletteItem {
    /// What the application calls it back by: the entry's variant id.
    pub key: String,
    pub title: String,
    /// Where it lives: "Mesh / Edge".
    pub path: String,
    /// One sentence, shown for the highlighted entry.
    pub detail: String,
    pub shortcut: String,
    /// Matched but not shown: the operator id, English names.
    pub keywords: String,
    /// Greyed and not runnable, the way the menu entry is.
    pub enabled: bool,
}

/// The palette's strings.
pub struct PaletteText {
    pub placeholder: String,
    pub nothing: String,
}

/// What the palette remembers.
#[derive(Clone, Debug, Default)]
pub struct PaletteState {
    pub open: bool,
    query: String,
    selected: usize,
    /// Focus the field on the frame after opening, once.
    focus: bool,
    /// Keys of the entries run from here, most recent first. Shown first when
    /// nothing is typed, which is Blender's arrangement: what you searched for
    /// last is what you are likeliest to search for again.
    pub recent: Vec<String>,
}

impl PaletteState {
    pub fn open(&mut self) {
        self.open = true;
        self.query.clear();
        self.selected = 0;
        self.focus = true;
    }

    /// Opens with something already typed — for a script, a test, or a
    /// button that means "search for this".
    pub fn open_with(&mut self, query: &str) {
        self.open();
        self.query = query.to_string();
    }

    fn remember(&mut self, key: &str) {
        self.recent.retain(|k| k != key);
        self.recent.insert(0, key.to_string());
        self.recent.truncate(10);
    }
}

/// How well one typed word matches `hay`, or `None` if it does not.
fn word_score(word: &str, hay: &str) -> Option<i32> {
    if let Some(at) = hay.find(word) {
        let starts_word = at == 0
            || hay[..at]
                .chars()
                .next_back()
                .is_some_and(|c| !c.is_alphanumeric());
        return Some(100 - (at as i32).min(50) + if starts_word { 40 } else { 0 });
    }
    // Scattered: every letter, in order.
    let mut letters = hay.chars();
    word.chars()
        .all(|c| letters.any(|h| h == c))
        .then_some(10)
}

/// How well `query` matches an entry, or `None` if some word is missing.
///
/// Words found in the title count double: "delete" should find Delete before
/// it finds an entry whose description mentions deleting.
pub fn score(query: &str, item: &PaletteItem) -> Option<i32> {
    let title = item.title.to_lowercase();
    let rest = format!("{} {}", item.path, item.keywords).to_lowercase();
    let mut total = 0;
    for word in query.to_lowercase().split_whitespace() {
        let in_title = word_score(word, &title).map(|s| s * 2);
        let elsewhere = word_score(word, &rest);
        total += in_title.max(elsewhere)?;
    }
    Some(total)
}

/// The entries to show, best first.
pub fn ranked(items: &[PaletteItem], query: &str, recent: &[String]) -> Vec<usize> {
    if query.trim().is_empty() {
        // Recent first, then the menus' own order.
        let mut out: Vec<usize> = recent
            .iter()
            .filter_map(|k| items.iter().position(|i| &i.key == k))
            .collect();
        for i in 0..items.len() {
            if !out.contains(&i) {
                out.push(i);
            }
        }
        return out;
    }
    let mut scored: Vec<(i32, usize)> = items
        .iter()
        .enumerate()
        .filter_map(|(i, item)| score(query, item).map(|s| (s, i)))
        .collect();
    // Best first; ties keep the menus' order, which is stable and learnable.
    scored.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    scored.into_iter().map(|(_, i)| i).collect()
}

/// Draws the palette if it is open. Returns the entry to run.
pub fn command_palette(
    ctx: &egui::Context,
    state: &mut PaletteState,
    items: &[PaletteItem],
    text: &PaletteText,
) -> Option<usize> {
    if !state.open {
        return None;
    }
    let order = ranked(items, &state.query, &state.recent);
    if state.selected >= order.len() {
        state.selected = order.len().saturating_sub(1);
    }
    let (up, down, enter, escape) = ctx.input(|i| {
        (
            i.key_pressed(Key::ArrowUp),
            i.key_pressed(Key::ArrowDown),
            i.key_pressed(Key::Enter),
            i.key_pressed(Key::Escape),
        )
    });
    if up {
        state.selected = state.selected.saturating_sub(1);
    }
    if down && state.selected + 1 < order.len() {
        state.selected += 1;
    }

    let mut chosen = None;
    let screen = ctx.content_rect();
    let width = (screen.width() - 32.0).clamp(240.0, 560.0);
    let area = egui::Area::new(egui::Id::new("command-palette"))
        .order(egui::Order::Foreground)
        .fixed_pos(egui::pos2(screen.center().x - width / 2.0, screen.min.y + 60.0))
        .show(ctx, |ui| {
            egui::Frame::popup(ui.style()).show(ui, |ui| {
                let theme = crate::theme::from_ui(ui);
                ui.set_width(width);
                let field = ui.add(
                    egui::TextEdit::singleline(&mut state.query)
                        .hint_text(&text.placeholder)
                        .desired_width(f32::INFINITY),
                );
                if state.focus {
                    field.request_focus();
                    state.focus = false;
                }
                if field.changed() {
                    state.selected = 0;
                }
                ui.separator();
                if order.is_empty() {
                    ui.weak(&text.nothing);
                    return;
                }
                egui::ScrollArea::vertical()
                    .max_height(360.0)
                    .auto_shrink([false, true])
                    .show(ui, |ui| {
                        for (row, &index) in order.iter().enumerate() {
                            let item = &items[index];
                            let highlighted = row == state.selected;
                            let response = ui
                                .add_enabled_ui(item.enabled, |ui| {
                                    ui.horizontal(|ui| {
                                        let label = if item.path.is_empty() {
                                            item.title.clone()
                                        } else {
                                            // A slash rather than `›`: the bundled
                                            // font has no glyph for the arrow, and a
                                            // box where a separator belongs reads as
                                            // a broken string.
                                            format!("{}  /  {}", item.path, item.title)
                                        };
                                        let r = ui.selectable_label(highlighted, label);
                                        ui.with_layout(
                                            egui::Layout::right_to_left(egui::Align::Center),
                                            |ui| {
                                                ui.label(
                                                    egui::RichText::new(&item.shortcut)
                                                        .color(theme.text_faint),
                                                );
                                            },
                                        );
                                        r
                                    })
                                    .inner
                                })
                                .inner;
                            if highlighted && (up || down) {
                                response.scroll_to_me(None);
                            }
                            if response.clicked() && item.enabled {
                                chosen = Some(index);
                            }
                        }
                    });
                if let Some(item) = order.get(state.selected).map(|i| &items[*i]) {
                    if !item.detail.is_empty() {
                        ui.separator();
                        ui.add(egui::Label::new(egui::RichText::new(&item.detail).color(theme.text_dim)).wrap());
                    }
                }
            });
        });

    if enter {
        if let Some(item) = order.get(state.selected).map(|i| &items[*i]) {
            if item.enabled {
                chosen = order.get(state.selected).copied();
            }
        }
    }
    let clicked_outside = ctx.input(|i| i.pointer.any_pressed())
        && ctx
            .input(|i| i.pointer.interact_pos())
            .is_some_and(|p| !area.response.rect.contains(p));
    if escape || clicked_outside || chosen.is_some() {
        state.open = false;
    }
    if let Some(index) = chosen {
        state.remember(&items[index].key);
    }
    chosen
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(key: &str, title: &str, path: &str, keywords: &str) -> PaletteItem {
        PaletteItem {
            key: key.into(),
            title: title.into(),
            path: path.into(),
            detail: String::new(),
            shortcut: String::new(),
            keywords: keywords.into(),
            enabled: true,
        }
    }

    fn items() -> Vec<PaletteItem> {
        vec![
            item("next", "Next Frame", "View", "view.next"),
            item("extrude", "Extrude", "Face", "mesh.extrude_region"),
            item("delete", "Delete", "Object", "object.delete"),
            item("bevel", "Bevel", "Edge", "mesh.bevel"),
            item("cube", "Cube", "Add / Mesh", "object.add_primitive"),
        ]
    }

    #[test]
    fn a_word_at_the_start_of_a_name_beats_one_inside_it() {
        let list = items();
        let order = ranked(&list, "ext", &[]);
        assert_eq!(list[order[0]].key, "extrude", "{order:?}");
        assert!(order.iter().any(|i| list[*i].key == "next"), "Next was dropped, not ranked");
    }

    #[test]
    fn every_word_has_to_be_there() {
        let list = items();
        let order = ranked(&list, "add cube", &[]);
        assert_eq!(order.len(), 1);
        assert_eq!(list[order[0]].key, "cube");
        assert!(ranked(&list, "add sphere", &[]).is_empty());
    }

    #[test]
    fn the_operator_id_is_searched_too() {
        // A Korean interface still answers to the English name a tutorial
        // uses, because the id is in the keywords.
        let mut list = items();
        list[1].title = "돌출".into();
        let order = ranked(&list, "extrude", &[]);
        assert_eq!(list[order[0]].key, "extrude");
    }

    #[test]
    fn scattered_letters_still_find_something_but_rank_last() {
        let list = items();
        let order = ranked(&list, "bvl", &[]);
        assert_eq!(list[order[0]].key, "bevel");
        assert!(score("bvl", &list[3]) < score("bev", &list[3]));
    }

    #[test]
    fn nothing_typed_lists_recent_first_then_the_menus_order() {
        let list = items();
        let order = ranked(&list, "", &["bevel".into(), "cube".into()]);
        let keys: Vec<&str> = order.iter().map(|i| list[*i].key.as_str()).collect();
        assert_eq!(keys, vec!["bevel", "cube", "next", "extrude", "delete"]);
    }

    #[test]
    fn remembering_moves_to_the_front_without_duplicates() {
        let mut state = PaletteState::default();
        for key in ["a", "b", "a"] {
            state.remember(key);
        }
        assert_eq!(state.recent, vec!["a".to_string(), "b".to_string()]);
        for i in 0..20 {
            state.remember(&i.to_string());
        }
        assert_eq!(state.recent.len(), 10, "the list grows without bound");
    }
}
