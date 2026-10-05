import { test } from "node:test";
import assert from "node:assert/strict";
import { allow, brevoPayload, parseInquiry, POST } from "../api/contact.ts";

test("a message with its details is accepted and trimmed", () => {
  const got = parseInquiry({ kind: "bug", message: "  It crashed  ", email: "a@b.kr", version: "0.2.0 (abc)", platform: "macos aarch64" });
  assert.deepEqual(got, { kind: "bug", message: "It crashed", name: undefined, email: "a@b.kr", version: "0.2.0 (abc)", platform: "macos aarch64" });
});

test("an unknown kind is just a message", () => {
  const got = parseInquiry({ kind: "spam", message: "hi" });
  assert.equal(typeof got === "object" && got.kind, "other");
});

test("empty, oversized, malformed and bot-filled messages are refused", () => {
  assert.equal(typeof parseInquiry({ message: "   " }), "string");
  assert.equal(typeof parseInquiry({ message: "x".repeat(5001) }), "string");
  assert.equal(typeof parseInquiry({ message: "hi", email: "not an address" }), "string");
  assert.equal(typeof parseInquiry({ message: "hi", name: 42 }), "string");
  assert.equal(parseInquiry({ message: "hi", website: "http://spam" }), "refused");
  assert.equal(typeof parseInquiry(null), "string");
});

test("the mail goes to the configured address and replies go to the writer", () => {
  const inquiry = parseInquiry({ kind: "feature", message: "Add a lathe\nplease", name: "Kim", email: "kim@x.kr" });
  assert.ok(typeof inquiry === "object");
  const p = brevoPayload(inquiry, "to@example.com", "from@example.com", "Kreate");
  assert.deepEqual(p.to, [{ email: "to@example.com" }]);
  assert.deepEqual(p.replyTo, { email: "kim@x.kr", name: "Kim" });
  assert.equal(p.subject, "[Kreate Feature request] Add a lathe");
  assert.match(p.textContent, /Add a lathe\nplease/);
});

test("without a reply address there is no replyTo", () => {
  const inquiry = parseInquiry({ message: "hello" });
  assert.ok(typeof inquiry === "object");
  assert.equal("replyTo" in brevoPayload(inquiry, "t@e.com", "f@e.com", "Kreate"), false);
});

test("one address is held to a few messages per window", () => {
  const t = 1_000_000;
  for (let i = 0; i < 5; i++) assert.equal(allow("1.2.3.4", t + i), true);
  assert.equal(allow("1.2.3.4", t + 10), false);
  assert.equal(allow("5.6.7.8", t + 10), true, "another address is not held back");
  assert.equal(allow("1.2.3.4", t + 11 * 60 * 1000), true, "and the window passes");
});

test("a request that does not say it is Kreate is refused, and the address never leaks", async () => {
  process.env.CONTACT_TO = "secret@example.com";
  const refused = await POST(new Request("http://x/api/contact", { method: "POST", body: "{}" }));
  assert.equal(refused.status, 403);
  // Not set up: says so without naming anything.
  delete process.env.BREVO_API_KEY;
  const unset = await POST(
    new Request("http://x/api/contact", {
      method: "POST",
      headers: { "x-kreate-client": "kreate/0.2.0", "x-forwarded-for": "9.9.9.9" },
      body: JSON.stringify({ message: "hi" }),
    }),
  );
  assert.equal(unset.status, 503);
  assert.doesNotMatch(await unset.text(), /secret@example\.com/);
});
