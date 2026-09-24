'use strict';
/*
 * Discord voice bridge for voice_translator.py.
 *
 * Started by the Python app (discord_source.py) as a child process, one per
 * running "Discord" session. It logs in as the configured bot, follows one
 * Discord user between voice channels, and receives every other speaker's
 * audio *separately* (Discord sends one stream per user, so there's no need
 * to guess who is talking). Each stream is decoded, down-mixed and
 * resampled to 16 kHz mono int16 here, then handed to Python.
 *
 * Why Node: voice receive has to decrypt Discord's end-to-end voice
 * encryption (DAVE). @discordjs/voice does that (via @snazzah/davey); the
 * Python libraries currently don't for received audio.
 *
 * Protocol with the parent process
 *   stdout: frames of [1 byte type][4 bytes big-endian length][payload]
 *     type 1 = JSON event (UTF-8)
 *     type 2 = audio: [8 bytes big-endian user id][int16 LE 16 kHz mono PCM]
 *   stdin:  one JSON command per line: {"type":"ignore","ids":[...]} or
 *           {"type":"shutdown"}. stdin closing also shuts down.
 *   stderr: human-readable log lines.
 * Nothing else may ever be written to stdout.
 *
 * Environment
 *   DISCORD_TOKEN      bot token (required)
 *   FOLLOW_USER_ID     the Discord user to follow into voice channels
 *   IGNORE_USER_IDS    comma/space separated user ids never transcribed
 *   BRIDGE_MODE        "follow" (default) or "check" (log in, report, exit)
 *   BRIDGE_FAKE_WAV    testing only: skip Discord, stream this 48 kHz mono
 *                      16-bit WAV as two alternating fake speakers
 */

const fs = require('fs');
const readline = require('readline');

const TOKEN = (process.env.DISCORD_TOKEN || '').trim();
const FOLLOW = (process.env.FOLLOW_USER_ID || '').trim();
const MODE = process.env.BRIDGE_MODE || 'follow';
let ignore = new Set(
  (process.env.IGNORE_USER_IDS || '').split(/[\s,;]+/).map(s => s.trim()).filter(Boolean)
);

// ── output framing ──────────────────────────────────────────────────────────
function writeFrame(type, payload) {
  const header = Buffer.alloc(5);
  header.writeUInt8(type, 0);
  header.writeUInt32BE(payload.length, 1);
  process.stdout.write(Buffer.concat([header, payload]));
}
function send(event) {
  writeFrame(1, Buffer.from(JSON.stringify(event), 'utf8'));
}
function sendAudio(userId, pcm16k) {
  const id = Buffer.alloc(8);
  id.writeBigUInt64BE(BigInt(userId), 0);
  writeFrame(2, Buffer.concat([id, pcm16k]));
}
function log(...args) {
  console.error('[discord-bridge]', ...args);
}

// ── 48 kHz stereo → 16 kHz mono ─────────────────────────────────────────────
// 48 000 / 16 000 is exactly 3, so each output sample is the average of three
// input frames (both channels) — a boxcar low-pass + decimation in one step.
class Downsampler {
  constructor() {
    this.carry = Buffer.alloc(0);
  }
  push(chunk) {
    const buf = this.carry.length ? Buffer.concat([this.carry, chunk]) : chunk;
    const frameBytes = 4; // stereo int16
    const groups = Math.floor(buf.length / (frameBytes * 3));
    const out = Buffer.alloc(groups * 2);
    for (let g = 0; g < groups; g++) {
      const base = g * frameBytes * 3;
      let sum = 0;
      for (let f = 0; f < 3; f++) {
        sum += buf.readInt16LE(base + f * 4) + buf.readInt16LE(base + f * 4 + 2);
      }
      out.writeInt16LE(Math.max(-32768, Math.min(32767, Math.round(sum / 6))), g * 2);
    }
    this.carry = Buffer.from(buf.subarray(groups * frameBytes * 3));
    return out;
  }
}

// ── lifecycle ───────────────────────────────────────────────────────────────
let shuttingDown = false;
const cleanups = [];
async function shutdown(code = 0) {
  if (shuttingDown) return;
  shuttingDown = true;
  for (const fn of cleanups.reverse()) {
    try { await fn(); } catch (e) { /* best effort */ }
  }
  // Give stdout a moment to flush the last frames.
  setTimeout(() => process.exit(code), 100);
}

// The parent closed our stdout (it stopped or crashed): nothing left to do.
process.stdout.on('error', () => process.exit(0));

const rl = readline.createInterface({ input: process.stdin });
rl.on('line', line => {
  let cmd;
  try { cmd = JSON.parse(line); } catch (e) { return; }
  if (cmd.type === 'ignore' && Array.isArray(cmd.ids)) {
    ignore = new Set(cmd.ids.map(String));
    log('ignore list updated:', [...ignore].join(', ') || '(empty)');
  } else if (cmd.type === 'shutdown') {
    shutdown(0);
  }
});
rl.on('close', () => shutdown(0)); // parent went away
process.on('SIGTERM', () => shutdown(0));
process.on('unhandledRejection', err => log('unhandled rejection:', err && err.stack || err));

// ── fake mode (tests without Discord) ───────────────────────────────────────
async function runFake(wavPath) {
  const wav = fs.readFileSync(wavPath);
  const dataStart = wav.indexOf('data') + 8;
  const mono48 = wav.subarray(dataStart);
  send({ type: 'ready', bot: 'FakeBot#0000', guilds: ['Fake Guild'] });
  send({ type: 'voice', state: 'joined', guild: 'Fake Guild', channel: 'general' });
  // BRIDGE_FAKE_SPEAKERS=N (default 2). With 2, they take turns every 3 s;
  // with more, everyone talks at once (each offset into the WAV) — a load
  // test for many simultaneous speakers.
  const count = Math.max(1, parseInt(process.env.BRIDGE_FAKE_SPEAKERS || '2', 10));
  const names = ['Alice', 'Bob', 'Carol', 'Dave', 'Erin', 'Frank', 'Grace', 'Heidi', 'Ivan', 'Judy'];
  const speakers = [];
  for (let i = 0; i < count; i++) {
    speakers.push({
      id: String(111111111111111111n * BigInt(i + 1)),
      name: names[i % names.length] + (i >= names.length ? ` ${Math.floor(i / names.length) + 1}` : ''),
      avatar: '',
      offset: Math.floor((i * mono48.length) / count / 2) * 2,
      ds: new Downsampler(),
    });
  }
  for (const s of speakers) send({ type: 'speaker', user_id: s.id, name: s.name, avatar: s.avatar });
  const frame = 960 * 2; // 20 ms of 48 kHz mono int16
  let tick = 0;
  const toStereo = chunk => {
    const stereo = Buffer.alloc(chunk.length * 2);
    for (let i = 0; i < chunk.length / 2; i++) {
      const v = chunk.readInt16LE(i * 2);
      stereo.writeInt16LE(v, i * 4);
      stereo.writeInt16LE(v, i * 4 + 2);
    }
    return stereo;
  };
  const timer = setInterval(() => {
    if (shuttingDown) return;
    const turn = Math.floor(tick / 150) % speakers.length;
    tick++;
    speakers.forEach((s, i) => {
      if (count <= 2 && i !== turn) return;
      if (ignore.has(s.id)) return;
      if (s.offset + frame > mono48.length) s.offset = 0;
      const chunk = mono48.subarray(s.offset, s.offset + frame);
      s.offset += frame;
      // Real Discord sends nothing while someone is silent — mimic that.
      let peak = 0;
      for (let k = 0; k < chunk.length; k += 2) peak = Math.max(peak, Math.abs(chunk.readInt16LE(k)));
      if (peak < 50) return;
      sendAudio(s.id, s.ds.push(toStereo(chunk)));
    });
  }, 20);
  cleanups.push(() => clearInterval(timer));
}

// ── real Discord ────────────────────────────────────────────────────────────
async function runDiscord() {
  const { Client, GatewayIntentBits, Events } = require('discord.js');
  const {
    joinVoiceChannel, entersState, VoiceConnectionStatus, EndBehaviorType, getVoiceConnection,
  } = require('@discordjs/voice');
  const prism = require('prism-media');

  if (!TOKEN) {
    send({ type: 'error', message: 'No Discord bot token set' });
    return shutdown(1);
  }

  const client = new Client({
    // Guilds + GuildVoiceStates only — no privileged intents needed. Members
    // in voice channels are cached from their voice states, and anyone else
    // is fetched individually over REST.
    intents: [GatewayIntentBits.Guilds, GatewayIntentBits.GuildVoiceStates],
  });
  cleanups.push(async () => {
    for (const guild of client.guilds.cache.values()) {
      const c = getVoiceConnection(guild.id);
      if (c) c.destroy();
    }
    await client.destroy();
  });

  let connection = null;
  let currentChannelId = null;
  let currentGuild = null;
  const subscriptions = new Map(); // userId -> { stream, decoder }
  const announced = new Set();

  async function announceSpeaker(guild, userId) {
    if (announced.has(userId)) return;
    announced.add(userId);
    let member = guild.members.cache.get(userId);
    if (!member) {
      try { member = await guild.members.fetch(userId); } catch (e) { member = null; }
    }
    send({
      type: 'speaker',
      user_id: userId,
      name: member ? member.displayName : `User ${userId.slice(-4)}`,
      avatar: member ? member.displayAvatarURL({ size: 128, extension: 'png' }) : '',
    });
  }

  function unsubscribe(userId) {
    const sub = subscriptions.get(userId);
    if (!sub) return;
    subscriptions.delete(userId);
    try { sub.stream.destroy(); } catch (e) {}
    try { sub.decoder.destroy(); } catch (e) {}
  }

  function subscribe(guild, userId) {
    if (subscriptions.has(userId) || ignore.has(userId) || userId === client.user.id) return;
    const opusStream = connection.receiver.subscribe(userId, {
      end: { behavior: EndBehaviorType.AfterSilence, duration: 1000 },
    });
    const decoder = new prism.opus.Decoder({ rate: 48000, channels: 2, frameSize: 960 });
    const ds = new Downsampler();
    subscriptions.set(userId, { stream: opusStream, decoder });
    announceSpeaker(guild, userId);
    opusStream.on('error', err => log(`stream error for ${userId}:`, err.message));
    decoder.on('error', err => log(`decode error for ${userId}:`, err.message));
    decoder.on('data', pcm => {
      if (ignore.has(userId)) return;
      const out = ds.push(pcm);
      if (out.length) sendAudio(userId, out);
    });
    opusStream.pipe(decoder);
    opusStream.once('end', () => unsubscribe(userId));
    opusStream.once('close', () => unsubscribe(userId));
  }

  function leave(reason) {
    for (const id of [...subscriptions.keys()]) unsubscribe(id);
    if (connection) {
      try { connection.destroy(); } catch (e) {}
    }
    connection = null;
    currentChannelId = null;
    currentGuild = null;
    send({ type: 'voice', state: 'left', reason });
  }

  async function join(channel) {
    if (currentChannelId === channel.id && connection) return;
    for (const id of [...subscriptions.keys()]) unsubscribe(id);
    if (connection) { try { connection.destroy(); } catch (e) {} }
    currentChannelId = channel.id;
    currentGuild = channel.guild;
    log(`joining #${channel.name} in ${channel.guild.name}`);
    const conn = joinVoiceChannel({
      channelId: channel.id,
      guildId: channel.guild.id,
      adapterCreator: channel.guild.voiceAdapterCreator,
      selfDeaf: false, // must hear to transcribe
      selfMute: true,
    });
    connection = conn;
    conn.on('error', err => log('voice connection error:', err.message));
    conn.on('debug', msg => { if (/dave/i.test(msg)) log('voice:', msg); });
    try {
      await entersState(conn, VoiceConnectionStatus.Ready, 20000);
    } catch (e) {
      if (connection === conn) {
        send({ type: 'error', message: `Could not connect to #${channel.name} (missing Connect permission?)` });
        leave('connect failed');
      }
      return;
    }
    if (connection !== conn) return;
    send({ type: 'voice', state: 'joined', guild: channel.guild.name, channel: channel.name });
    conn.receiver.speaking.on('start', userId => {
      if (connection === conn) subscribe(channel.guild, userId);
    });
    conn.on(VoiceConnectionStatus.Disconnected, async () => {
      // Moved/kicked or a network hiccup: give it a moment to reconnect.
      try {
        await Promise.race([
          entersState(conn, VoiceConnectionStatus.Signalling, 5000),
          entersState(conn, VoiceConnectionStatus.Connecting, 5000),
        ]);
      } catch (e) {
        if (connection === conn) leave('disconnected from the voice channel');
      }
    });
  }

  function findFollowedChannel() {
    for (const guild of client.guilds.cache.values()) {
      const vs = guild.voiceStates.cache.get(FOLLOW);
      if (vs && vs.channel) return vs.channel;
    }
    return null;
  }

  client.once(Events.ClientReady, async () => {
    const guilds = [...client.guilds.cache.values()].map(g => g.name);
    send({ type: 'ready', bot: client.user.tag, bot_id: client.user.id, guilds });
    log(`logged in as ${client.user.tag} (${guilds.length} servers)`);
    const channel = FOLLOW ? findFollowedChannel() : null;
    if (MODE === 'check') {
      send({
        type: 'check',
        bot: client.user.tag,
        bot_id: client.user.id,
        guilds,
        follow_channel: channel ? `#${channel.name} (${channel.guild.name})` : null,
      });
      return shutdown(0);
    }
    if (!channel) {
      send({ type: 'voice', state: 'not_in_voice' });
      return;
    }
    await join(channel);
  });

  client.on(Events.VoiceStateUpdate, async (oldState, newState) => {
    if (MODE === 'check') return;
    if (newState.id === FOLLOW) {
      if (newState.channel && newState.channelId !== currentChannelId) {
        await join(newState.channel);
      } else if (!newState.channelId && oldState.channelId) {
        leave('you left the voice channel');
      }
    } else if (newState.id === client.user.id && !newState.channelId && currentChannelId) {
      leave('the bot was disconnected from the voice channel');
    }
  });

  client.on(Events.Error, err => log('client error:', err.message));

  try {
    await client.login(TOKEN);
  } catch (e) {
    send({ type: 'error', message: `Discord login failed: ${e.message}` });
    return shutdown(1);
  }
}

if (process.env.BRIDGE_FAKE_WAV) {
  runFake(process.env.BRIDGE_FAKE_WAV);
} else {
  runDiscord();
}
