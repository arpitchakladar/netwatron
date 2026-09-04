{
  description = "Development flake for netwatron";
  inputs = {
    nixpkgs.url = "nixpkgs";
    devenv = {
      url = "github:cachix/devenv";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };
  nixConfig = {
    extra-substituters = [
      "https://cache.nixos.org"
      "https://nix-community.cachix.org"
      "https://devenv.cachix.org"
      "https://cuda-maintainers.cachix.org"
    ];
    extra-trusted-public-keys = [
      "cache.nixos.org-1:6NCHdD59X431o0gWypbMrAURkbJ16ZPMQFGspcDShjY="
      "nix-community.cachix.org-1:mB9FSh9qf2dCimDSUo8Zy7bkq5CX+/rkCWyvRCYg3Fs="
      "devenv.cachix.org-1:w1cLUi8dv3hnoSPGAuibQv+f9TZLr6cv/Hm9XgU50cw="
      "cuda-maintainers.cachix.org-1:0dq3bujKpuEPMCX6U4WylrUDZ9JyUG0VpVZa7CNfq5E="
    ];
  };
  outputs =
    {
      nixpkgs,
      devenv,
      ...
    }@inputs:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
          cudaSupport = true;
          cudaCapabilities = [ "12.0" ];
          cudaForwardCompat = false;
        };
        overlays = [
          (final: prev: {
            cudaPackages = prev.cudaPackages_13;
            ucx = prev.ucx.override { enableCuda = false; };
            openmpi = prev.openmpi.override { cudaSupport = false; };
          })
        ];
      };
    in
    {
      formatter.${system} = pkgs.nixfmt-tree;
      devShells.${system}.default = devenv.lib.mkShell {
        inherit inputs pkgs;
        modules = [
          (
            { ... }:
            {
              git-hooks.hooks = {
                nixfmt.enable = true;
                black.enable = true;
                pyright.enable = true;
              };
              env.LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [ pkgs.libpcap ];
              languages = {
                nix = {
                  enable = true;
                  lsp = {
                    enable = true;
                    package = pkgs.nixd;
                  };
                };
                shell = {
                  enable = true;
                  lsp = {
                    enable = true;
                    package = pkgs.bash-language-server;
                  };
                };
                python = {
                  enable = true;
                  package = pkgs.python3.withPackages (
                    ps: with ps; [
                      scapy
                      dpkt
                      cryptography
                      black
                      textual
                      rich
                      numpy
                      pandas
                      scikit-learn
                      ((ps.torch-bin.override { cudaPackages = pkgs.cudaPackages_13; }).overrideAttrs (old: {
                        pythonRelaxDeps = [ "setuptools" ];
                        nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ ps.pythonRelaxDepsHook ];
                      }))
                    ]
                  );
                  lsp = {
                    enable = true;
                    package = pkgs.pyright;
                  };
                };
              };
              packages = [
                pkgs.libpcap
              ];
            }
          )
        ];
      };
    };
}
