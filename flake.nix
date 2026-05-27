{
  description = "Media classifier for Jellyfin library organization";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  };

  outputs = {
    self,
    nixpkgs,
  }: let
    supportedSystems = ["x86_64-linux" "aarch64-linux"];
    forAllSystems = nixpkgs.lib.genAttrs supportedSystems;
  in {
    packages = forAllSystems (system: let
      pkgs = nixpkgs.legacyPackages.${system};
      classifierPython = pkgs.python3.withPackages (ps: [ps.anitopy ps.rapidfuzz]);
    in {
      default = pkgs.stdenv.mkDerivation {
        pname = "media-classifier";
        version = "2.0.0";
        src = ./.;

        nativeBuildInputs = [pkgs.makeWrapper];

        installPhase = ''
          mkdir -p $out/bin $out/lib
          cp media-classifier.py $out/lib/
          cp model_gym.py $out/lib/

          makeWrapper ${classifierPython}/bin/python3 $out/bin/media-classifier \
            --add-flags "$out/lib/media-classifier.py" \
            --prefix PATH : ${pkgs.lib.makeBinPath [pkgs.ffmpeg]}

          makeWrapper ${classifierPython}/bin/python3 $out/bin/model-gym \
            --add-flags "$out/lib/model_gym.py" \
            --prefix PATH : ${pkgs.lib.makeBinPath [pkgs.ffmpeg]}
        '';
      };
    });

    nixosModules.default = import ./module.nix {inherit self;};
  };
}
