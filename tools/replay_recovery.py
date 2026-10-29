"""Persistent, bounded source retries; failed sources never become successes."""
import json
import time
from pathlib import Path


class RecoveryQueue:
    def __init__(self, path, attempts=3, delay=30):
        self.path = Path(path)
        self.attempts = attempts
        self.delay = delay
        self.records = json.loads(self.path.read_text()) if self.path.exists() else {}

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.records, indent=2) + '\n')
        temporary.replace(self.path)

    def failed(self, task, stage, error, now=None):
        now = time.time() if now is None else now
        record = self.records.setdefault(str(task), {'history': []})
        record['history'].append({'stage': stage, 'error': repr(error), 'time': now})
        count = sum(row['stage'] == stage for row in record['history'])
        record.update(status='quarantined' if count >= self.attempts else 'retrying',
                      next_retry=now + self.delay * 2 ** min(count - 1, 5), stage=stage)
        self.save()
        return record['status']

    def ready(self, task, now=None):
        record = self.records.get(str(task), {})
        return record.get('status') != 'quarantined' and record.get('next_retry', 0) <= (time.time() if now is None else now)

    def completed(self, task):
        if str(task) in self.records:
            self.records[str(task)]['status'] = 'recovered'
            self.records[str(task)]['next_retry'] = 0
            self.save()

    def quarantined(self):
        return [int(task) for task, record in self.records.items() if record.get('status') == 'quarantined']
