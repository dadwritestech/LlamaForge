"""Create the first-run config for a bundled or existing llama.cpp checkout."""
import argparse, json, os


def _server_bin(build_dir, windows=None, isfile=os.path.isfile):
    candidates = [
        os.path.join(build_dir, "bin", "Release", "llama-server.exe"),
        os.path.join(build_dir, "bin", "llama-server.exe"),
        os.path.join(build_dir, "bin", "llama-server"),
    ]
    for path in candidates:
        if isfile(path):
            return path
    windows = os.name == "nt" if windows is None else windows
    return candidates[0] if windows else candidates[-1]


def build(repo_root, llama_src=None, windows=None):
    repo_root = os.path.abspath(os.path.expanduser(repo_root))
    llama_src = llama_src or os.path.join(repo_root, "llama.cpp")
    llama_src = os.path.abspath(os.path.expanduser(llama_src))
    build_dir = os.path.join(llama_src, "build")
    return {
        "llama_src": llama_src,
        "build_dir": build_dir,
        "server_bin": _server_bin(build_dir, windows=windows),
        "models_ini": os.path.join(repo_root, "models.ini"),
        "model_dirs": [],
        "router_port": 8080,
        "panel_port": 8090,
        "router_host": "127.0.0.1",
        "router_api_key": "",
        "cmake_flags": {},
        "git_remote": "https://github.com/ggml-org/llama.cpp",
    }


def write(path, repo_root, llama_src=None):
    cfg = build(repo_root, llama_src)
    with open(path, "x", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")
    return cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--repo-root", required=True)
    p.add_argument("--llama-src")
    args = p.parse_args()
    write(args.config, args.repo_root, args.llama_src)


if __name__ == "__main__":
    main()
