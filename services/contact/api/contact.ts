// Kreate's contact relay: Help > Contact posts here, and this sends it on to
// the developers through Brevo's transactional email API.
//
// Why a relay at all: the application is handed to anyone who downloads it,
// so whatever it carries can be read out of it. The Brevo key and the address
// the messages go to live here, in the deployment's environment, and nowhere
// else — not in the application, not in the repository.
//
// Environment:
//   BREVO_API_KEY      the Brevo API key (Settings > SMTP & API > API keys)
//   CONTACT_TO         where messages go
//   CONTACT_FROM       a sender Brevo has verified for this account
//   CONTACT_FROM_NAME  the sender's name (optional, "Kreate")

// The one Node global used here, declared rather than pulling in @types/node
// for it.
declare const process: { env: Record<string, string | undefined> };

/** What the application sends. */
export interface Inquiry {
  kind: "bug" | "feature" | "question" | "other";
  message: string;
  name?: string;
  email?: string;
  version?: string;
  platform?: string;
}

const KINDS: Record<Inquiry["kind"], string> = {
  bug: "Bug",
  feature: "Feature request",
  question: "Question",
  other: "Message",
};

const LIMITS = { message: 5000, name: 100, email: 200, version: 60, platform: 60 } as const;

/** Reads and checks a request body. A string is why it was refused. */
export function parseInquiry(body: unknown): Inquiry | string {
  if (typeof body !== "object" || body === null) return "the body must be a JSON object";
  const b = body as Record<string, unknown>;
  // A field no person fills in: the form never shows it, a form-filling bot
  // does not know that.
  if (typeof b.website === "string" && b.website.trim() !== "") return "refused";
  const text = (key: keyof typeof LIMITS): string | undefined | null => {
    const v = b[key];
    if (v === undefined || v === null || v === "") return undefined;
    if (typeof v !== "string") return null;
    const t = v.trim();
    return t.length > LIMITS[key] ? null : t;
  };
  const kind = typeof b.kind === "string" && b.kind in KINDS ? (b.kind as Inquiry["kind"]) : "other";
  const message = text("message");
  if (message === null) return `the message is longer than ${LIMITS.message} characters`;
  if (!message) return "the message is empty";
  const fields = { name: text("name"), email: text("email"), version: text("version"), platform: text("platform") };
  for (const [key, value] of Object.entries(fields)) {
    if (value === null) return `${key} is too long or not text`;
  }
  if (fields.email && !/^[^\s@<>]+@[^\s@<>]+\.[^\s@<>]+$/.test(fields.email)) return "the reply address is not an email address";
  return {
    kind,
    message,
    name: fields.name ?? undefined,
    email: fields.email ?? undefined,
    version: fields.version ?? undefined,
    platform: fields.platform ?? undefined,
  };
}

/** The Brevo request for an inquiry. */
export function brevoPayload(inquiry: Inquiry, to: string, from: string, fromName: string) {
  const firstLine = inquiry.message.split("\n")[0].slice(0, 60);
  const lines = [
    inquiry.message,
    "",
    "—",
    `Kind: ${KINDS[inquiry.kind]}`,
    `From: ${inquiry.name ?? "(no name)"} <${inquiry.email ?? "no reply address"}>`,
    `Kreate: ${inquiry.version ?? "unknown"} on ${inquiry.platform ?? "unknown"}`,
  ];
  return {
    sender: { name: fromName, email: from },
    to: [{ email: to }],
    // Replying answers the person who wrote, when they left an address.
    ...(inquiry.email ? { replyTo: { email: inquiry.email, ...(inquiry.name ? { name: inquiry.name } : {}) } } : {}),
    subject: `[Kreate ${KINDS[inquiry.kind]}] ${firstLine}`,
    textContent: lines.join("\n"),
  };
}

// A few messages a minute from one address is a person; more is not. Kept in
// the instance's memory, so it is a speed bump rather than a wall — an
// instance that scales out or restarts starts counting again.
const WINDOW_MS = 10 * 60 * 1000;
const PER_WINDOW = 5;
const recent = new Map<string, number[]>();

export function allow(key: string, now = Date.now()): boolean {
  const times = (recent.get(key) ?? []).filter((t) => now - t < WINDOW_MS);
  if (times.length >= PER_WINDOW) {
    recent.set(key, times);
    return false;
  }
  times.push(now);
  recent.set(key, times);
  return true;
}

const json = (status: number, body: unknown) =>
  new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });

export async function GET() {
  return json(200, { ok: true, service: "kreate-contact" });
}

export async function POST(request: Request) {
  // The application says who it is. Anyone can say it, so this keeps out
  // passing scanners, not a determined sender.
  if (!(request.headers.get("x-kreate-client") ?? "").startsWith("kreate/")) {
    return json(403, { ok: false, error: "unknown client" });
  }
  const length = Number(request.headers.get("content-length") ?? "0");
  if (length > 20_000) return json(413, { ok: false, error: "the message is too large" });

  const ip = (request.headers.get("x-forwarded-for") ?? "").split(",")[0].trim() || "unknown";
  if (!allow(ip)) return json(429, { ok: false, error: "too many messages; try again in a few minutes" });

  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return json(400, { ok: false, error: "the body is not JSON" });
  }
  const inquiry = parseInquiry(body);
  if (typeof inquiry === "string") return json(400, { ok: false, error: inquiry });

  const { BREVO_API_KEY, CONTACT_TO, CONTACT_FROM, CONTACT_FROM_NAME } = process.env;
  if (!BREVO_API_KEY || !CONTACT_TO || !CONTACT_FROM) {
    console.error("contact: BREVO_API_KEY, CONTACT_TO and CONTACT_FROM must all be set");
    return json(503, { ok: false, error: "the contact service is not set up" });
  }

  const sent = await fetch("https://api.brevo.com/v3/smtp/email", {
    method: "POST",
    headers: { "api-key": BREVO_API_KEY, "content-type": "application/json", accept: "application/json" },
    body: JSON.stringify(brevoPayload(inquiry, CONTACT_TO, CONTACT_FROM, CONTACT_FROM_NAME || "Kreate")),
    signal: AbortSignal.timeout(10_000),
  }).catch((e: unknown) => e as Error);

  if (sent instanceof Error || !sent.ok) {
    // Logged in full here; the sender is told only that it did not go.
    const detail = sent instanceof Error ? sent.message : `${sent.status} ${await sent.text()}`;
    console.error("contact: Brevo refused", detail);
    return json(502, { ok: false, error: "the message could not be sent; please try again later" });
  }
  return json(200, { ok: true });
}
