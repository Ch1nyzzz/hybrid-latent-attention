"""Keep a local copy of trisol training curves (no wandb: trisol nodes cannot reach it).

For each job: if its output model is registered, download the per-rank event logs
(rank-*.jsonl, eval-*.json; never weights); otherwise rebuild rank-0.jsonl from the
job's stdout, where rank 0 prints every emitted event. Canceled jobs register no
output model and their logs expire, so pull soon after a job ends (or while it runs).

    python -m hla.trisol.pull_curves JOB_ID [JOB_ID ...]
    python -m hla.trisol.pull_curves --since 2026-09-21 --search loop-s6
Output: results/latent/training-curves/<job>/ with source.json describing where it came from.
"""
import argparse, json, subprocess, time
from datetime import datetime, timedelta
from pathlib import Path

WINDOW = timedelta(hours=6)          # longer history windows fail with window_too_large
LIMIT = 5000                         # per-query line limit of the history API
FILES = ('rank-*.jsonl', 'eval-*.json', '*summary*.json', 'complete.json')


def trisol(*args, timeout=300):
    """Retry on the history API's rate limit (10 queries/min per user)."""
    for attempt in range(6):
        r = subprocess.run(['trisol', *args], capture_output=True, text=True, timeout=timeout)
        if 'rate_limit' not in r.stdout + r.stderr:
            time.sleep(7)
            return r
        time.sleep(15 * (attempt + 1))
    return r


def stamp(text):
    return datetime.fromisoformat(text.replace('Z', '+00:00'))


def rfc3339(t):
    return t.strftime('%Y-%m-%dT%H:%M:%SZ')  # the history API rejects fractional seconds


def from_model(job, out):
    name = job['name']
    listing = trisol('model', 'versions', name, '-o', 'json')
    if listing.returncode:
        return None
    data = json.loads(listing.stdout or '{}')
    versions = data if isinstance(data, list) else data.get('items', [])
    if not versions:
        return None
    code = max(int(v.get('version_code') or v.get('code') or 0) for v in versions)
    args = ['model', 'download', f'{name}:{code}', '-o', f'{out}/']
    for pattern in FILES:
        args += ['--include', pattern]
    r = trisol(*args, timeout=1800)
    if r.returncode:
        raise RuntimeError(f'{name}:{code} download failed: {r.stderr.strip()[-300:]}')
    return dict(source='output-model', model=f'{name}:{code}')


def log_events(job_id, since, until):
    """rank-0 event rows in [since, until); halves the window when the 5000-line query limit is hit."""
    r = trisol('train', 'logs', job_id, '--history', '--grep', '"event"', '--since', rfc3339(since),
               '--until', rfc3339(until), '--direction', 'forward', '--tail', str(LIMIT))
    if r.returncode:
        raise RuntimeError(f'{job_id} logs {rfc3339(since)}..{rfc3339(until)}: {r.stderr.strip()[-300:]}')
    lines = r.stdout.splitlines()
    if len(lines) >= LIMIT and until - since > timedelta(minutes=2):
        middle = since + (until - since) / 2
        return log_events(job_id, since, middle) + log_events(job_id, middle, until)
    rows = []
    for line in lines:
        i = line.find('{"event": ')
        if i < 0:
            continue
        try:
            row = json.loads(line[i:])
        except json.JSONDecodeError:
            continue
        if row.get('rank', 0) == 0:
            rows.append(row)
    return rows


def from_logs(job, out):
    start, end = stamp(job['created_at']), stamp(job.get('finished_at') or job['updated_at']) + timedelta(minutes=5)
    rows, t = [], start
    while t < end:
        u = min(t + WINDOW, end)
        rows += log_events(job['id'], t, u)
        t = u
    if not rows:
        return None
    (out / 'rank-0.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return dict(source='job-stdout', events=len(rows), updates=sum(r['event'] == 'update' for r in rows))


def pull(job_id, root):
    job = json.loads(trisol('train', 'get', job_id, '-o', 'json').stdout)
    out = root / job_id
    out.mkdir(parents=True, exist_ok=True)
    meta = from_model(job, out) or from_logs(job, out) or dict(source='none')
    meta.update(job=job_id, name=job['name'], status=job['status'], created_at=job['created_at'],
                pulled_at=datetime.now().astimezone().isoformat(timespec='seconds'))
    (out / 'source.json').write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('jobs', nargs='*')
    p.add_argument('--since', help='also pull every job created on/after this date (YYYY-MM-DD)')
    p.add_argument('--search', default='loop-s6', help='job-name substring for --since')
    p.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[2] / 'results/latent/training-curves')
    a = p.parse_args()
    jobs = list(a.jobs)
    if a.since:
        listing = json.loads(trisol('train', 'list', '--search', a.search, '--all', '-o', 'json').stdout)
        items = listing if isinstance(listing, list) else listing.get('items', [])
        jobs += [j['id'] for j in items if j['created_at'] >= a.since and j['id'] not in jobs]
    for job_id in jobs:
        pull(job_id, a.output)


if __name__ == '__main__':
    main()
