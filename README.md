# media-classifier

Scans source directories and sorts media into a Jellyfin-style library
tree (Movies / TV Shows / Anime) via symlinks. Pipeline: anitopy +
regex fast-path, AniList/Wikipedia/ffprobe evidence, scoring, then an
arbiter for ambiguous cases — the Jev (TypeSafe System One) decision
model via OpenRouter when configured, else a local Ollama LLM. Optionally
triggers a Jellyfin library rescan and cleans broken symlinks daily.

## Options (`services.media-classifier`)

| option | type | default | note |
|---|---|---|---|
| `enable` | bool | `false` | |
| `sourceDirs` | [str] | — | dirs to scan |
| `mediaBase` | str | `/srv/media` | symlink target tree |
| `categories` | attrs str | `{movie="Movies"; tv="TV Shows"; anime="Anime";}` | |
| `categoryOverrides` | attrs str | `{}` | pin a show to `movie`/`tv`/`anime` |
| `ollamaHost` | str | `http://localhost:11434` | |
| `ollamaModel` | str | `qwen3:0.6b` | fallback arbiter model |
| `jevApiKeyFile` | str | `""` | file with an OpenRouter key; empty disables Jev |
| `jevModel` | str | `typesafe/jev-1.13` | OpenRouter System One model |
| `jevUrl` | str | `https://openrouter.ai/api/v1/systemone` | |
| `jellyfinApiKey` | str | `""` | empty = no rescan |
| `jellyfinUrl` | str | `http://localhost:8096` | |
| `user` / `group` | str | `root` | |
| `timer.enable` | bool | `false` | periodic run |
| `timer.onCalendar` | str | `*-*-* 0/6:00:00` | every 6h |
| `symlinkCleanup.enable` | bool | `true` | daily broken-symlink sweep |

## Arbiters

Ambiguous files are handed to one of two arbiters, in order:

1. **Jev** (when `jevApiKeyFile` is set). [Jev](https://openrouter.ai/typesafe/jev-1.13)
   is a TypeSafe "System One" decision model served through OpenRouter. It
   takes the parsed filename signals plus ffprobe context as `state` and a
   typed `choice` question, and returns exactly one of `anime`/`tv`/`movie`
   with a calibrated confidence — no prose to parse and no way to get an
   unknown category back. The key is read from the file at runtime so it
   never enters the Nix store; point `jevApiKeyFile` at an agenix secret.
2. **Ollama** (fallback, and used when no Jev key is configured).

Fast-path (structural) and high-confidence scored results bypass the
arbiters. A medium-confidence scored result is overridden by Jev when a key
is configured.

## Usage

```nix
inputs.media-classifier.url = "github:mccartykim/media-classifier";

imports = [ inputs.media-classifier.nixosModules.default ];
services.media-classifier = {
  enable = true;
  sourceDirs = [ "/srv/incoming" ];
  timer.enable = true;
  jellyfinApiKey = "...";
  # Optional: use Jev as the arbiter (key in an agenix secret).
  jevApiKeyFile = config.age.secrets.openrouter-api-key.path;
};
```

CLI also runnable directly: `media-classifier --config <json>`. There's
a `model-gym` binary for tuning the classifier against labelled samples.

## Contents

`media-classifier.py` (classifier), `model_gym.py` (tuning harness),
`module.nix` (NixOS module), `flake.nix`, `aliases.example.json`,
`test_media_classifier.py` + `conftest.py` (tests).