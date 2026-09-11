"""Check exactly the staged Git content, without printing secret values."""
import argparse
from pathlib import PurePosixPath
import re
import subprocess
import sys

SECRET_PATTERNS = {
    "private key": r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----",
    "GitHub credential": r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b",
    "provider credential": r"\b(?:sk-(?:proj-|or-v1-)?[A-Za-z0-9_-]{24,}|gsk_[A-Za-z0-9]{30,}|AIza[A-Za-z0-9_-]{30,}|hf_[A-Za-z0-9]{25,})\b",
    "AWS access key": r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
    "credential in URL": r"https?://[^\s/:]+:[^\s/@]+@",
    "Windows user path": r"[A-Za-z]:[\\/]+Users[\\/]+[^\s\\/]+",
}
COMPILED = {name: re.compile(pattern) for name, pattern in SECRET_PATTERNS.items()}


def prohibited_path(name):
    path = PurePosixPath(name)
    parts = path.parts
    base = path.name.lower()
    if any(p in {'.venv', 'node_modules', '.codex', '.agents', '__pycache__'} for p in parts):
        return True
    if base.startswith('.env') and name != '.env.example':
        return True
    if path.suffix.lower() in {'.db', '.sqlite', '.sqlite3', '.pem', '.key', '.p12', '.jks', '.bak', '.log', '.apk'}:
        return True
    if parts[0] in {'training', 'identity', 'thread_memory', '.artifacts', 'screenshots'}:
        return True
    if parts[0] == 'data' and name not in {
        'data/selectors/screen_signatures.json', 'data/templates/reply_templates.json'
    }:
        return True
    return base.startswith(('tmp_', 'tmp-')) or 'whatsapp chat with' in base


def inspect_content(name, data):
    findings = []
    if prohibited_path(name):
        findings.append(f'{name}: private/runtime path')
    if len(data) > 2_000_000:
        findings.append(f'{name}: file exceeds public source size limit')
    if b'\0' in data:
        if not name.startswith('docs/images/'):
            findings.append(f'{name}: unexpected binary asset')
        return findings
    text = data.decode('utf-8', errors='replace')
    for label, pattern in COMPILED.items():
        for match in pattern.finditer(text):
            line = text.count('\n', 0, match.start()) + 1
            findings.append(f'{name}:{line}: {label}')
    # UK mobile fixtures must use the reserved drama range.
    for match in re.finditer(r'(?<!\w)(?:\+44[ -]?7\d{3}[ -]?\d{6}|07\d{3}[ -]?\d{6})(?!\d)', text):
        digits = re.sub(r'\D', '', match.group())
        if not (digits.startswith('447700900') or digits.startswith('07700900')):
            findings.append(f'{name}:{text.count(chr(10), 0, match.start()) + 1}: non-example mobile number')
    return findings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision', help='Scan a committed tree instead of the index')
    args = parser.parse_args()
    listing = ['git', 'ls-tree', '-rz', '--name-only', args.revision] if args.revision else ['git', 'ls-files', '-z']
    names = subprocess.check_output(listing).decode().strip('\0').split('\0')
    names = [name for name in names if name]
    if not names:
        raise SystemExit('No staged files to audit')
    findings = []
    for name in names:
        ref = f'{args.revision}:{name}' if args.revision else f':{name}'
        data = subprocess.check_output(['git', 'show', ref])
        findings.extend(inspect_content(name, data))
    for finding in findings:
        print(finding)
    print(f'Public release audit: {len(names)} files, {len(findings)} findings.')
    return bool(findings)


if __name__ == '__main__':
    sys.exit(main())
