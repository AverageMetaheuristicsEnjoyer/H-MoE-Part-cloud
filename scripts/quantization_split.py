"""Reserve unstarted original run paths so active workers finish without duplicates."""
import datetime
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
campaign = Path('/workspace-SR006.nfs3/hmoe-cloud/quantization-20261001')
plan = json.loads((root / 'configs/quantization-split-20261001.json').read_text())
result = []
for row in plan:
    path = Path(row['old_run_dir'])
    if not path.is_relative_to(campaign) or not path.parent.is_dir():
        raise RuntimeError('Unexpected source run path')
    reservation = dict(row, action='reserved_for_independent_job',
                       utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    try:
        with path.open('x') as stream:
            json.dump(reservation, stream)
            stream.write('\n')
    except FileExistsError:
        if path.is_dir():
            reservation = dict(row, action='already_started_or_completed')
        else:
            reservation = json.loads(path.read_text())
            if reservation.get('old_run_dir') != str(path):
                raise RuntimeError('Unexpected existing reservation')
    result.append(reservation)
    print('SPLIT_CELL ' + json.dumps(reservation), flush=True)
(campaign / 'split-pending.json').write_text(json.dumps(result, indent=2) + '\n')
print('SPLIT_COMPLETE=' + str(sum(r['action']=='reserved_for_independent_job' for r in result)), flush=True)
