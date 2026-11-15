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

    # llama.cpp keeps converter-specific Python dependencies separately.
    converter_requirements = directory / "requirements" / "requirements-convert_hf_to_gguf.txt"
    if converter_requirements.is_file():
        run([sys.executable, "-m", "pip", "install", "-r", str(converter_requirements)])
    else:
        requirements = directory / "requirements.txt"
        if requirements.is_file():
            run([sys.executable, "-m", "pip", "install", "-r", str(requirements)])

    return directory


def upload_release_asset(asset: Path, tag: str) -> None:
    import os
    import http.client
    import urllib.error
    import urllib.parse

    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("Set GH_TOKEN (or GITHUB_TOKEN) before using --upload.")

    api = "https://api.github.com/repos/CrazeStudio/CrazeAi"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "CrazeAI-GGUF-Exporter",
    }

    def request(url: str, method: str = "GET", data: bytes | None = None,
                content_type: str = "application/json"):
        req_headers = dict(headers)
        req_headers["Content-Type"] = content_type
        req = urllib.request.Request(url, data=data, headers=req_headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=180) as response:
                body = response.read()
                return response.status, json.loads(body) if body else {}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"GitHub API {exc.code}: {detail}") from exc

    _, release = request(f"{api}/releases/tags/{urllib.parse.quote(tag, safe='')}")
    release_id = release["id"]
    asset_name = asset.name

    for existing in release.get("assets", []):
        if existing.get("name") == asset_name:
            request(f"{api}/releases/assets/{existing['id']}", method="DELETE")
            print(f"Replaced existing release asset: {asset_name}", flush=True)

    upload_path = (
        f"/repos/CrazeStudio/CrazeAi/releases/{release_id}/assets"
        f"?name={urllib.parse.quote(asset_name)}"
    )
    size = asset.stat().st_size
    if size > 2 * 1024 * 1024 * 1024:
        raise RuntimeError("GitHub Release assets must be smaller than 2 GiB.")
    print(f"Uploading {asset_name} ({size / (1024**2):.1f} MiB) to GitHub Release {tag}", flush=True)

    # Stream from disk instead of reading a potentially multi-gigabyte GGUF into RAM.
    connection = http.client.HTTPSConnection("uploads.github.com", timeout=300)
    connection.putrequest("POST", upload_path)
    connection.putheader("Authorization", f"Bearer {token}")
    connection.putheader("Accept", "application/vnd.github+json")
    connection.putheader("X-GitHub-Api-Version", "2022-11-28")
    connection.putheader("User-Agent", "CrazeAI-GGUF-Exporter")
    connection.putheader("Content-Type", "application/octet-stream")
    connection.putheader("Content-Length", str(size))
    connection.endheaders()
    with asset.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            connection.send(chunk)
    response = connection.getresponse()
    response_body = response.read().decode("utf-8", errors="replace")
    connection.close()
    if response.status not in (200, 201):
        raise RuntimeError(f"GitHub asset upload failed ({response.status}): {response_body}")
    print(f"Release asset uploaded: https://github.com/CrazeStudio/CrazeAi/releases/tag/{tag}", flush=True)


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
    parser.add_argument(
        "--upload", action="store_true",
        help="Upload the finished GGUF as an asset to the existing GitHub Release.",
    )
    parser.add_argument("--release-tag", default="v0.1.0-test")
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
    if not final_path.is_file() or final_path.stat().st_size == 0:
        raise RuntimeError("GGUF output is missing or empty; refusing to upload.")
    if args.upload:
        upload_release_asset(final_path, args.release_tag)


if __name__ == "__main__":
    main()
