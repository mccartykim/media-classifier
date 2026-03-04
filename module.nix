{self}: {
  config,
  lib,
  pkgs,
  ...
}: let
  cfg = config.services.media-classifier;

  configFile = pkgs.writeText "media-classifier-config.json" (builtins.toJSON {
    sourceDirs = cfg.sourceDirs;
    mediaBase = cfg.mediaBase;
    categories = cfg.categories;
    ollamaHost = cfg.ollamaHost;
    ollamaModel = cfg.ollamaModel;
    ffprobePath = "${pkgs.ffmpeg}/bin/ffprobe";
  });
in {
  options.services.media-classifier = {
    enable = lib.mkEnableOption "media classifier for Jellyfin library organization";

    package = lib.mkOption {
      type = lib.types.package;
      default = self.packages.${pkgs.system}.default;
      description = "The media-classifier package to use.";
    };

    sourceDirs = lib.mkOption {
      type = lib.types.listOf lib.types.str;
      description = "Source directories to scan for media files.";
    };

    mediaBase = lib.mkOption {
      type = lib.types.str;
      default = "/srv/media";
      description = "Base directory for categorized media symlinks.";
    };

    categories = lib.mkOption {
      type = lib.types.attrsOf lib.types.str;
      default = {
        movie = "Movies";
        tv = "TV Shows";
        anime = "Anime";
      };
      description = "Mapping of category keys to directory names under mediaBase.";
    };

    ollamaHost = lib.mkOption {
      type = lib.types.str;
      default = "http://localhost:11434";
      description = "Ollama API host URL for LLM arbiter.";
    };

    ollamaModel = lib.mkOption {
      type = lib.types.str;
      default = "qwen3:0.6b";
      description = "Ollama model to use for classification.";
    };

    jellyfinApiKey = lib.mkOption {
      type = lib.types.str;
      default = "";
      description = "Jellyfin API key for triggering library rescans after classification. If empty, no rescan is triggered.";
    };

    jellyfinUrl = lib.mkOption {
      type = lib.types.str;
      default = "http://localhost:8096";
      description = "Jellyfin server URL for triggering library rescans.";
    };

    user = lib.mkOption {
      type = lib.types.str;
      default = "root";
      description = "User to run the classifier as.";
    };

    group = lib.mkOption {
      type = lib.types.str;
      default = "root";
      description = "Group to run the classifier as.";
    };

    timer = {
      enable = lib.mkOption {
        type = lib.types.bool;
        default = false;
        description = "Enable periodic timer for the classifier.";
      };

      onCalendar = lib.mkOption {
        type = lib.types.str;
        default = "*-*-* 0/6:00:00";
        description = "Systemd calendar expression for the timer.";
      };
    };

    symlinkCleanup = {
      enable = lib.mkOption {
        type = lib.types.bool;
        default = true;
        description = "Enable periodic cleanup of broken symlinks in media directories.";
      };
    };
  };

  config = lib.mkIf cfg.enable {
    # Media directory structure
    systemd.tmpfiles.rules = let
      base = cfg.mediaBase;
    in
      ["d ${base} 2775 ${cfg.user} ${cfg.group} -"]
      ++ (lib.mapAttrsToList (
        _: dirName: ''d "${base}/${dirName}" 2775 ${cfg.user} ${cfg.group} -''
      )
      cfg.categories);

    # Main classifier service
    systemd.services.media-classifier = {
      description = "Classify media and create Jellyfin symlinks";
      after = ["network-online.target"];
      wants = ["network-online.target"];
      serviceConfig = {
        Type = "oneshot";
        User = cfg.user;
        Group = cfg.group;
        UMask = "0002";
        StateDirectory = "media-classifier";
        ExecStart = "${cfg.package}/bin/media-classifier --config ${configFile}";
      } // lib.optionalAttrs (cfg.jellyfinApiKey != "") {
        ExecStartPost = pkgs.writeShellScript "trigger-jellyfin-scan" ''
          ${pkgs.curl}/bin/curl -sf -X POST \
            "${cfg.jellyfinUrl}/Library/Refresh?api_key=${cfg.jellyfinApiKey}" \
            || echo "Warning: Jellyfin scan trigger failed (non-fatal)"
        '';
      };
    };

    # Optional timer for classifier
    systemd.timers.media-classifier = lib.mkIf cfg.timer.enable {
      wantedBy = ["timers.target"];
      timerConfig = {
        OnCalendar = cfg.timer.onCalendar;
        RandomizedDelaySec = "30m";
        Persistent = true;
      };
    };

    # Broken symlink cleanup
    systemd.services.media-symlink-cleanup = lib.mkIf cfg.symlinkCleanup.enable {
      description = "Remove broken symlinks from Jellyfin media directories";
      serviceConfig = {
        Type = "oneshot";
        User = cfg.user;
        Group = cfg.group;
        ExecStart = pkgs.writeShellScript "cleanup-symlinks" ''
          ${pkgs.findutils}/bin/find ${cfg.mediaBase} -xtype l -delete -print
        '';
      };
    };

    systemd.timers.media-symlink-cleanup = lib.mkIf cfg.symlinkCleanup.enable {
      wantedBy = ["timers.target"];
      timerConfig = {
        OnCalendar = "daily";
        Persistent = true;
      };
    };
  };
}
