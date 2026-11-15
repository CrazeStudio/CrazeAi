from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

DEFAULT_ADAPTER_URL = (
    "https://github.com/CrazeStudio/CrazeAi/releases/download/"
    "v0.1.0-test/CrazeAI-Test-LoRA.zip"
)
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def run(command: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def download_adapter(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.is_file():
        print(f"Downloading adapter: {url}", flush=True)
        request = urllib.request.Request(
            url, headers={"User-Agent": "CrazeAI-GGUF-Exporter/1.0"}
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            with destination.open("wb") as output:
                shutil.copyfileobj(response, output)

    adapter_dir = destination.parent / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    required = {"adapter_config.json", "adapter_model.safetensors"}

    with zipfile.ZipFile(destination) as archive:
        names = {Path(name).name: name for name in archive.namelist()}
        missing = required - names.keys()
        if missing:
            raise RuntimeError(
                "Release ZIP is missing required LoRA files: "
                + ", ".join(sorted(missing))
            )
        for filename in required:
            with archive.open(names[filename]) as source:
                with (adapter_dir / filename).open("wb") as target:
                    shutil.copyfileobj(source, target)

    return adapter_dir


def prepare_llama_cpp(directory: Path) -> Path:
    if not (directory / "convert_hf_to_gguf.py").is_file():
        if directory.exists():
            shutil.rmtree(directory)
        run([
            "git", "clone", "--depth", "1",
            "https://github.com/ggml-org/llama.cpp.git", str(directory)
        ])

    requirements = directory / "requirements.txt"
    if requirements.is_file():
        run([sys.executable, "-m", "pip", "install", "-r", str(requirements)])

    return directory


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the CrazeAI LoRA release, merge it, and export GGUF."
    )
    parser.add_argument("--adapter-url", default=DEFAULT_ADAPTER_URL)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gguf"))
    parser.add_argument(
        "--quantization",
        default="Q4_K_M",
        choices=["Q4_K_M", "Q5_K_M", "Q8_0", "F16"],
        help="GGUF output type. F16 skips the quantization/build step.",
    )
    parser.add_argument(
        "--llama-cpp-dir", type=Path, default=Path("tools/llama.cpp")
    )
    parser.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="Keep the merged Transformers model and temporary F16 GGUF.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    output_dir = (root / args.output_dir).resolve()
    llama_dir = (root / args.llama_cpp_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    zip_path = output_dir / "CrazeAI-Test-LoRA.zip"
    adapter_dir = download_adapter(args.adapter_url, zip_path)
    adapter_config = json.loads(
        (adapter_dir / "adapter_config.json").read_text(encoding="utf-8")
    )
    adapter_base = adapter_config.get("base_model_name_or_path")
    if adapter_base and adapter_base != BASE_MODEL:
        raise RuntimeError(
            f"Unexpected adapter base model: {adapter_base}; expected {BASE_MODEL}"
        )

    merged_dir = output_dir / "merged-transformers"
    run([
        sys.executable, "-m", "scripts.export_model",
        "--model", str(adapter_dir),
        "--output", str(merged_dir),
        "--merge-adapter",
    ], cwd=root)

    llama_dir = prepare_llama_cpp(llama_dir)
    f16_path = output_dir / "CrazeAI-Test-F16.gguf"
    run([
        sys.executable, str(llama_dir / "convert_hf_to_gguf.py"),
        str(merged_dir), "--outfile", str(f16_path), "--outtype", "f16",
    ], cwd=llama_dir)

    if args.quantization == "F16":
        final_path = f16_path
    else:
        run([
            "cmake", "-S", str(llama_dir), "-B", str(llama_dir / "build"),
            "-DGGML_CUDA=OFF",
        ])
        run([
            "cmake", "--build", str(llama_dir / "build"),
            "--config", "Release", "--parallel", "2",
            "--target", "llama-quantize",
        ])
        quantizer = llama_dir / "build" / "bin" / "llama-quantize"
        if not quantizer.is_file():
            raise FileNotFoundError(f"Quantizer was not built: {quantizer}")
        final_path = output_dir / f"CrazeAI-Test-{args.quantization}.gguf"
        run([
            str(quantizer), str(f16_path), str(final_path), args.quantization
        ])

    if not args.keep_intermediates and args.quantization != "F16":
        shutil.rmtree(merged_dir, ignore_errors=True)
        f16_path.unlink(missing_ok=True)

    print(f"GGUF export complete: {final_path}", flush=True)


if __name__ == "__main__":
    main()
