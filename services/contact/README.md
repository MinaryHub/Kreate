# kreate-contact

The relay behind Kreate's **Help ▸ Contact the Developers…**. The application
posts a message here; this sends it on through [Brevo](https://www.brevo.com)'s
transactional email API.

It exists so that neither the Brevo key nor the address messages reach is in
the application or in this repository. Both live in the deployment's
environment and nowhere else.

Deployed on Vercel as the project `kreate-contact`, at
`https://kreate-contact.vercel.app/api/contact` — the address Kreate is built
with unless `KREATE_CONTACT_URL` says otherwise.

## Environment

| Variable | What |
|---|---|
| `BREVO_API_KEY` | A Brevo API key: Brevo ▸ Settings ▸ SMTP & API ▸ API Keys. |
| `CONTACT_TO` | Where messages go. |
| `CONTACT_FROM` | The sender. Brevo only sends from a sender it has verified: Brevo ▸ Senders, Domains & Dedicated IPs. |
| `CONTACT_FROM_NAME` | The sender's name. Optional; `Kreate`. |

Set them for Production, then redeploy:

```sh
cd services/contact
vercel env add BREVO_API_KEY production
vercel env add CONTACT_FROM production
vercel deploy --prod
```

Until all three of `BREVO_API_KEY`, `CONTACT_TO` and `CONTACT_FROM` are set,
the relay answers `503 the contact service is not set up` and Kreate shows
that under the form.

## What it accepts

`POST /api/contact` with `x-kreate-client: kreate/<version>` and a JSON body:

```json
{ "kind": "bug | feature | question | other", "message": "…",
  "name": "optional", "email": "optional, becomes Reply-To",
  "version": "optional", "platform": "optional" }
```

It refuses a request without the client header, a message that is empty or
longer than 5000 characters, a malformed reply address, and the hidden
`website` field a form-filling bot fills in. One address may send five
messages in ten minutes; the count is kept per instance, so it slows a flood
rather than stopping one. The answer is `{"ok": true}` or
`{"ok": false, "error": "…"}`, and never contains the address.

## Tests

```sh
npm test        # node --test; Node 24 runs the TypeScript as is
```
