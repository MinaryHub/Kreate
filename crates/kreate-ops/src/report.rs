//! A line an operator leaves about what it did, for whoever ran it to show:
//! how many problems a check found, how many elements matched. Not part of
//! the history — a sentence for the moment, read once.

use std::cell::RefCell;

thread_local! {
    static LAST: RefCell<Option<String>> = const { RefCell::new(None) };
}

/// Says `what` about the operation that is running.
pub fn say(what: impl Into<String>) {
    LAST.with(|l| *l.borrow_mut() = Some(what.into()));
}

/// What the last operation said, once; nothing if it said nothing.
pub fn take() -> Option<String> {
    LAST.with(|l| l.borrow_mut().take())
}
