"""Check portable shell defaults without launching training or touching local config."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def find_bash():
    if os.name == "nt":
        git = shutil.which("git")
        if git:
            candidate = Path(git).resolve().parent.parent / "bin" / "bash.exe"
            if candidate.is_file():
                return str(candidate)
        return None
    return shutil.which("bash")


@unittest.skipUnless(find_bash(), "Bash is required to validate server configuration")
class ServerConfigurationTests(unittest.TestCase):
    def source_config(self, root, overrides=None):
        env = dict(os.environ)
        for key in list(env):
            if key.upper().endswith("PROXY") or key in {"OUT_DIR", "TOKENIZER_NAME_OR_PATH", "TRAIN_DATA_PATH", "VAL_DATA_PATH", "TEST_DATA_PATH", "MEMMAP_OUT_DIR"}:
                env.pop(key)
        env.update(overrides or {})
        result = subprocess.run([find_bash(), "-c", 'set -eu; source "$1"; printf "%s\\n" "$HTTPS_PROXY" "$TOKENIZER_NAME_OR_PATH" "$OUT_DIR" "$TRAIN_DATA_PATH"', "test", str(root / "scripts/server/server_env.sh")], env=env, capture_output=True, text=True, check=True)
        return result.stdout.splitlines()

    def fixture(self, root):
        dest = root / "scripts/server"
        dest.mkdir(parents=True)
        (root / "env").mkdir()
        shutil.copyfile(ROOT / "scripts/server/server_env.sh", dest / "server_env.sh")

    def test_defaults_have_no_private_proxy_or_tokenizer(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root)
            proxy, tokenizer, out, train = self.source_config(root)
            self.assertEqual(proxy, "")
            self.assertEqual(tokenizer, "")
            self.assertTrue(out.endswith("/rosa_runs"))
            self.assertTrue(train.endswith("/data/minipile/train-*.jsonl"))

    def test_explicit_environment_overrides_defaults(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root)
            values = self.source_config(root, {"HTTPS_PROXY": "http://localhost:9000", "TOKENIZER_NAME_OR_PATH": "/custom/tokenizer", "OUT_DIR": "/custom/output", "TRAIN_DATA_PATH": "/custom/train.jsonl"})
            self.assertEqual(values, ["http://localhost:9000", "/custom/tokenizer", "/custom/output", "/custom/train.jsonl"])

    def test_local_config_loads_and_preserves_explicit_environment(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root)
            (root / "env/server.local.sh").write_text('export TOKENIZER_NAME_OR_PATH="${TOKENIZER_NAME_OR_PATH:-/local/tokenizer}"\nexport OUT_DIR="${OUT_DIR:-/local/output}"\n', encoding="utf-8")
            self.assertEqual(self.source_config(root)[1:3], ["/local/tokenizer", "/local/output"])
            self.assertEqual(self.source_config(root, {"TOKENIZER_NAME_OR_PATH": "/explicit"})[1], "/explicit")

    def test_all_server_and_compatibility_scripts_parse(self):
        for script in (ROOT / "scripts").rglob("*.sh"):
            with self.subTest(script=script.name):
                subprocess.run([find_bash(), "-n", str(script)], check=True, capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()
