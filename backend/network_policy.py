"""Pure router network/key policy. No config or process side effects."""
import argparse, copy, json, re, secrets, sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

STRONG_KEY_RE = re.compile(r"^[A-Za-z0-9._~-]{32,256}$")
LEGACY_KEY_RE = re.compile(r"^[\x21-\x7E]{1,256}$")
PUBLIC_CONFIG_KEYS = (
    "theme", "cvd", "auto_load_model", "vram_bandwidths", "presets",
    "preset_bindings", "active_engine",
)


@dataclass(frozen=True)
class Assessment:
    host: str
    access_scope: str
    configured_security_status: str
    key_status: str
    has_api_key: bool
    remediation_required: bool
    start_allowed: bool
    message: str

    def public(self):
        return asdict(self)


@dataclass(frozen=True)
class NetworkMutation:
    router_host: str
    router_api_key: str
    access_scope: str
    key_action: str
    assessment: Assessment
    generated_api_key: str | None = None


def key_status(value):
    if value is None or value == "":
        return "absent"
    if not isinstance(value, str):
        return "invalid"
    if STRONG_KEY_RE.fullmatch(value):
        return "strong"
    if LEGACY_KEY_RE.fullmatch(value):
        return "legacy"
    return "invalid"


def assess(host, key):
    clean_host = host if isinstance(host, str) else ""
    ks = key_status(key)
    has_key = isinstance(key, str) and bool(key)
    if clean_host == "127.0.0.1" and ks == "absent":
        return Assessment(clean_host, "local", "local", ks, False,
                          False, True, "")
    if clean_host == "127.0.0.1" and ks in ("strong", "legacy"):
        return Assessment(clean_host, "local", "local_keyed", ks, has_key,
                          False, True, "")
    if clean_host == "0.0.0.0" and ks in ("strong", "legacy"):
        status = "protected" if ks == "strong" else "protected_legacy"
        message = "" if ks == "strong" else "Rotate this legacy API key when convenient."
        return Assessment(clean_host, "lan", status, ks, True,
                          False, True, message)
    message = ("LAN router start refused: select local access or configure a "
               "usable API key before sharing the router.")
    return Assessment(clean_host, "legacy", "unsafe_legacy", ks, has_key,
                      True, False, message)


def generate_key():
    key = secrets.token_urlsafe(32)
    if key_status(key) != "strong":
        raise RuntimeError("generated API key did not satisfy policy")
    return key


def start_error(host, key):
    result = assess(host, key)
    return "" if result.start_allowed else result.message


def _request_parts(body):
    if not isinstance(body, Mapping):
        raise ValueError("network request must be an object")
    has_new = "access_scope" in body
    has_old = "host" in body
    if has_new == has_old:
        raise ValueError("provide access_scope or the deprecated host shape, not both")
    if has_new:
        extra = set(body) - {"access_scope", "key_action", "api_key"}
        if extra:
            raise ValueError("unsupported network fields: " + ", ".join(sorted(extra)))
        scope = body.get("access_scope")
        action = body.get("key_action", "keep")
        supplied = body.get("api_key") if "api_key" in body else None
        supplied_present = "api_key" in body
    else:
        extra = set(body) - {"host", "api_key"}
        if extra:
            raise ValueError("unsupported network fields: " + ", ".join(sorted(extra)))
        host = body.get("host")
        if host not in ("127.0.0.1", "0.0.0.0"):
            raise ValueError("host must be 127.0.0.1 or 0.0.0.0")
        scope = "local" if host == "127.0.0.1" else "lan"
        supplied_present = "api_key" in body
        supplied = body.get("api_key") if supplied_present else None
        action = "keep" if not supplied_present else ("clear" if supplied == "" else "replace")
    return scope, action, supplied, supplied_present


def apply_request(current, body):
    scope, action, supplied, supplied_present = _request_parts(body)
    if scope not in ("local", "lan"):
        raise ValueError("access_scope must be local or lan")
    if action not in ("keep", "generate", "replace", "clear"):
        raise ValueError("key_action must be keep, generate, replace, or clear")
    if action == "replace" and not supplied_present:
        raise ValueError("api_key is required for replace")
    if action != "replace" and supplied_present:
        if not (action == "clear" and supplied == ""):
            raise ValueError("api_key is valid only with key_action replace")

    current_key = current.get("router_api_key", "") if isinstance(current, Mapping) else ""
    generated = None
    if action == "keep":
        selected = current_key
    elif action == "generate":
        selected = generated = generate_key()
    elif action == "replace":
        if key_status(supplied) != "strong":
            raise ValueError("replacement API key must be a strong 32-256 character URL-safe token")
        selected = supplied
    else:
        if scope != "local":
            raise ValueError("an API key can be cleared only for local access")
        selected = ""

    host = "127.0.0.1" if scope == "local" else "0.0.0.0"
    result = assess(host, selected)
    if not result.start_allowed:
        raise ValueError(result.message)
    return NetworkMutation(host, selected, scope, action, result, generated)


def public_config(cfg):
    out = {key: copy.deepcopy(cfg[key]) for key in PUBLIC_CONFIG_KEYS if key in cfg}
    out["router_api_key_configured"] = bool(cfg.get("router_api_key"))
    return out


def preflight_config_file(path):
    try:
        with open(path, encoding="utf-8-sig") as f:
            cfg = json.load(f)
        if not isinstance(cfg, dict):
            raise ValueError("config root must be an object")
    except Exception as exc:
        return False, "Router start refused: config could not be read (%s)." % exc
    reason = start_error(cfg.get("router_host", "127.0.0.1"),
                         cfg.get("router_api_key", ""))
    return (not reason, reason)


def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", metavar="CONFIG")
    args = parser.parse_args(argv)
    if not args.preflight:
        parser.error("--preflight CONFIG is required")
    ok, message = preflight_config_file(args.preflight)
    if not ok:
        print(message, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
