{
  description = "DwarfStar 4 – DeepSeek V4 Flash inference engine";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.rocmSupport = pkgs.lib.elem system [
            "x86_64-linux" "aarch64-linux"
          ];
        };

        # Common native build inputs for all variants
        commonNativeInputs = with pkgs; [
          gnumake coreutils bison which
        ];

        # Build a variant that just delegates to a Makefile target
        makeDS4 = target: extraEnv: pkgs.stdenv.mkDerivation {
          pname = "ds4";
          version = "0.1.0";
          src = ./.;
          nativeBuildInputs = commonNativeInputs;
          buildPhase = ''
            make ${target} ${toString (pkgs.lib.concatStringsSep " " (
              pkgs.lib.mapAttrsToList (n: v: "${n}=${toString v}") extraEnv
            ))}
          '';
          installPhase = ''
            mkdir -p $out/bin
            for bin in ds4 ds4-server ds4-bench ds4-eval ds4-agent; do
              [ -x "$bin" ] && install -m 755 "$bin" "$out/bin/"
            done
          '';
          meta.homepage = "https://github.com/antirez/ds4";
          meta.license = pkgs.lib.licenses.mit;
        };

        rocmPkgs = pkgs.rocmPackages;

        # ROCm arch – override via --rocm-arch on command line if desired
        rocmArch = "gfx1151";
      in {
        packages = {
          # ---------- macOS / Metal ----------
          ds4-metal = makeDS4 "all" { };

          # ---------- Linux CUDA ----------
          ds4-cuda = pkgs.stdenv.mkDerivation {
            pname = "ds4-cuda";
            version = "0.1.0";
            src = ./.;

            nativeBuildInputs = commonNativeInputs ++ [ pkgs.gcc pkgs.cudaPackages.cuda_nvcc ];
            buildInputs = with pkgs; [
              cudaPackages.cuda_cudart
              cudaPackages.libcublas
            ];
            # The Makefile expects NVCC and CUDA_LDLIBS; set them explicitly
            # since CUDA packages are in different store paths, not a single CUDA_HOME.
            NVCC = "${pkgs.cudaPackages.cuda_nvcc}/bin/nvcc";
            CUDA_LDLIBS = "-lm -Xcompiler -pthread -L${pkgs.cudaPackages.cuda_cudart}/lib -L${pkgs.cudaPackages.libcublas}/lib -lcudart -lcublas";
            # DGX Spark / GB10 uses Blackwell GPU → sm_120 for __dp4a support
            CUDA_ARCH = "sm_120";
            buildPhase = ''
              make cuda CUDA_ARCH="$CUDA_ARCH" NVCC="$NVCC" CUDA_LDLIBS="$CUDA_LDLIBS" CC="${pkgs.gcc}/bin/gcc"
            '';

            installPhase = ''
              mkdir -p $out/bin
              for bin in ds4 ds4-server ds4-bench ds4-eval ds4-agent; do
                [ -x "$bin" ] && install -m 755 "$bin" "$out/bin/"
              done
            '';

            meta.homepage = "https://github.com/antirez/ds4";
            meta.license = pkgs.lib.licenses.mit;
          };

          # ---------- Linux ROCm (the working one) ----------
          ds4-rocm = pkgs.stdenv.mkDerivation {
            pname = "ds4-rocm";
            version = "0.1.0";
            src = ./.;

            nativeBuildInputs = commonNativeInputs ++ [
              pkgs.gcc
              rocmPkgs.clr               # hipcc
            ];

            buildInputs = [
              rocmPkgs.clr
              rocmPkgs.hipblas
              rocmPkgs."hipblas-common"
              rocmPkgs.rocwmma
              rocmPkgs.hipcub
              rocmPkgs.rocprim
            ];

            ROCM_PATH = "${rocmPkgs.clr}";
            ROCM_ARCH = rocmArch;
            GPU_BACKEND = "rocm";
            CC = "${pkgs.gcc}/bin/gcc";
            HIP_PATH = "${rocmPkgs.clr}";
            HIP_PLATFORM = "amd";
            HIP_CLANG_PATH = "${rocmPkgs.clr}/bin";
            DEVICE_LIB_PATH = "${rocmPkgs.clr}/amdgcn/bitcode";
            NATIVE_CPU_FLAG = "-march=x86-64";

            LD_LIBRARY_PATH = pkgs.lib.concatStringsSep ":" [
              "${rocmPkgs.hipblas}/lib"
              "${rocmPkgs.clr}/lib"
            ];

            buildPhase = ''
              make rocm ROCM_ARCH="${rocmArch}" \
                CC="$CC" \
                GPU_BACKEND=rocm \
                ROCM_PATH="$ROCM_PATH" \
                NATIVE_CPU_FLAG="$NATIVE_CPU_FLAG" \
                GPU_LDLIBS="-lm -pthread -L${rocmPkgs.hipblas}/lib -lhipblas" \
                GPU_CFLAGS="-O3 -ffast-math -fno-finite-math-only -pthread -Wno-unused-command-line-argument --offload-arch=${rocmArch} -I${rocmPkgs.rocwmma}/include -I${rocmPkgs.hipcub}/include -I${rocmPkgs.hipblas}/include -I${rocmPkgs."hipblas-common"}/include -I${rocmPkgs.rocprim}/include"
            '';

            installPhase = ''
              mkdir -p $out/bin
              for bin in ds4 ds4-server ds4-bench ds4-eval ds4-agent ds4_test; do
                [ -x "$bin" ] && install -m 755 "$bin" "$out/bin/"
              done
            '';

            meta.description = "DwarfStar 4 (ROCm ${rocmArch})";
            meta.homepage = "https://github.com/antirez/ds4";
            meta.license = pkgs.lib.licenses.mit;
            meta.platforms = pkgs.lib.platforms.linux;
          };

          # ---------- CPU-only (diagnostics) ----------
          ds4-cpu = makeDS4 "cpu" { };
        };

        # Platform-appropriate default
        defaultPackage = self.packages.${system}.ds4-metal;

        # Dev shell with all tools needed for any platform
        devShell = pkgs.mkShell {
          name = "ds4-dev";
          buildInputs = with pkgs; [
            gcc
            gnumake
            coreutils
            bison
            which
          ] ++ pkgs.lib.optionals pkgs.stdenv.isLinux [
            rocmPkgs.clr
            rocmPkgs.hipblas
            rocmPkgs."hipblas-common"
            rocmPkgs.rocwmma
            rocmPkgs.hipcub
            rocmPkgs.rocprim
          ];
          ROCM_PATH = if pkgs.stdenv.isLinux then "${rocmPkgs.clr}" else "";
          ROCM_ARCH = rocmArch;
          GPU_BACKEND = if pkgs.stdenv.isLinux then "rocm" else "";
          HIP_PATH = if pkgs.stdenv.isLinux then "${rocmPkgs.clr}" else "";
          HIP_PLATFORM = if pkgs.stdenv.isLinux then "amd" else "";
        };
      });
}
