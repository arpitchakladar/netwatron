{
  description = "Home Manager configuration of arpit.";
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
    ];
    extra-trusted-public-keys = [
      "cache.nixos.org-1:6NCHdD59X431o0gWypbMrAURkbJ16ZPMQFGspcDShjY="
      "nix-community.cachix.org-1:mB9FSh9qf2dCimDSUo8Zy7bkq5CX+/rkCWyvRCYg3Fs="
      "devenv.cachix.org-1:w1cLUi8dv3hnoSPGAuibQv+f9TZLr6cv/Hm9XgU50cw="
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
      pkgs = nixpkgs.legacyPackages.${system};
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

              # Expose libpcap to Python and the rest of the shell
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
