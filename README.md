# media-classifier

Scans source directories and sorts media into a Jellyfin-style library
tree (Movies / TV Shows / Anime) via symlinks. Pipeline: anitopy +
regex fast-path, AniList/Wikipedia/ffprobe evidence, scoring, then a
local Ollama LLM as arbiter for ambiguous cases. Optionally triggers a
Jellyfin library rescan and cleans broken symlinks daily.

## Options (`services.media-classifier`)

| option | type | default | note |
|---|---|---|---|
| `enable` | bool | `false` | |
| `sourceDirs` | [str] | — | dirs to scan |
| `mediaBase` | str | `/srv/media` | symlink target tree |
| `categories` | attrs str | `{movie="Movies"; tv="TV Shows"; anime="Anime";}` | |
| `ollamaHost` | str | `http://localhost:11434` | |
| `ollamaModel` | str | `qwen3:0.6b` | arbiter model |
| `jellyfinApiKey` | str | `""` | empty = no rescan |
| `jellyfinUrl` | str | `http://localhost:8096` | |
| `user` / `group` | str | `root` | |
| `timer.enable` | bool | `false` | periodic run |
| `timer.onCalendar` | str | `*-*-* 0/6:00:00` | every 6h |
| `symlinkCleanup.enable` | bool | `true` | daily broken-symlink sweep |

## Usage

```nix
inputs.media-classifier.url = "github:mccartykim/media-classifier";

imports = [ inputs.media-classifier.nixosModules.default ];
services.media-classifier = {
  enable = true;
  sourceDirs = [ "/srv/incoming" ];
  timer.enable = true;
  jellyfinApiKey = "...";
};
```

CLI also runnable directly: `media-classifier --config <json>`. There's
a `model-gym` binary for tuning the classifier against labelled samples.

## Contents

`media-classifier.py` (classifier), `model_gym.py` (tuning harness),
`module.nix` (NixOS module), `flake.nix`, `aliases.example.json`,
`test_media_classifier.py` + `conftest.py` (tests).