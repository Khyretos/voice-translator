# Discord voice channel captions

Pick **Audio Source → Discord Voice Channel** and a bot joins the voice
channel you're in and captions **every speaker separately**, each with their
Discord avatar and server nickname. Discord sends each person's audio as its
own stream, so the app knows exactly who said what; nothing is guessed from
a mixed recording. In OBS the popout shows one row per person talking,
stacked vertically, each fading out on its own.

This is an extra audio source: the other sources (server device, browser
mic, browser tab) are unchanged. A Discord session doesn't use a mic. Your
own voice comes through Discord like everyone else's (add your own ID to the
ignore list if you don't want it).

## One-time setup

1. **Create the bot.** Go to <https://discord.com/developers/applications>,
   click **New Application**, open **Bot**, and click **Reset Token** to get the
   token. No privileged intents are needed.
2. **Give the app the token.** Either paste it into **Bot token** in the
   Discord settings of the web page (saved with the session, like every other
   setting), or set it once for all sessions in `docker-compose.yml`:
   ```yaml
   environment:
     - DISCORD_BOT_TOKEN=your-token-here
   ```
   Don't share the token with anyone; it controls the bot.
3. **Invite the bot to your server(s).** Press **🔍 Check bot**: it logs
   in, lists the servers the bot is in, and prints an invite link with the
   permissions it needs (View Channel, Connect, Speak). Inviting needs
   *Manage Server* on that server.
4. **Enter your Discord user ID.** In Discord go to Settings → Advanced →
   Developer Mode, then right-click yourself and choose **Copy User ID**.

## Using it

- Join a voice channel on a server the bot is in, then press **▶️ Start**.
  The status shows `Listening to Discord: #channel (server)`.
- The bot follows you: switch channels and it switches too. Leave voice and
  the session stops, like the auto-stop for a lost mic.
- **Ignore these user IDs**: people who are never captioned (for example
  your own ID, or a music bot). Changes apply immediately.
- **Avatar side** (left/right), **Show names**, **Max speakers on screen**,
  plus the usual font, colour, outline, alignment and **Vertical position**
  settings all apply to the rows. Open popouts update themselves when you
  change these.
- Translation uses the session's target language for everyone.

Works with **Whisper** (each speaker gets their own live pipeline with
partial captions, same as the mic source) and **Vosk**. Moonshine isn't
supported for this source.

## Limits

- **Servers only.** Bots can't join direct-message or group calls, so this
  only works in servers the bot has been invited to.
- **One voice channel per bot per server.** Two sessions using the same bot
  can't follow people into different channels on the *same* server at the
  same time; use a second bot for that.
- **Whisper load grows with speakers.** Everyone talking at once means one
  request stream per person. The per-speaker workers never queue up old
  audio: if the server can't keep up, speech is merged into fewer, larger
  requests, and after ~12 s of backlog the oldest audio is skipped, so
  captions stay current instead of drifting behind. With 20 simultaneous
  (simulated) speakers the app itself used about a fifth of one CPU core.
- **People can see the bot in the channel.** It's transcribing everyone
  there, so let the people you talk with know. Discord's developer policy
  expects that.

## How it works

`discord_bridge/bridge.js` is a small Node.js program, started by the app
for each running Discord session and stopped with it. Node is used because
receiving voice means decrypting Discord's end-to-end voice encryption
(DAVE). `@discordjs/voice` supports that, while the Python Discord libraries
currently don't for *received* audio (py-cord 2.8 marks voice receive as
broken because of DAVE). The bridge decodes each speaker's audio to 16 kHz
mono and passes it to Python over a pipe (no extra ports or services).
Python runs one VAD + Whisper worker per speaker
(`discord_pipeline.py`) and shows the results as one row per person
(`subtitles.SpeakerBoard`).

The Docker image includes Node and the bridge's packages. For a non-Docker
install you need Node.js 22.12+ and `npm ci` in `discord_bridge/`.
