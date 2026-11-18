"""Prepare the original project's modeling inputs from the public v1 Kaggle CSVs."""
import argparse
import csv
import hashlib
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE_HASHES = {
    'fraudTrain.csv': 'fd7139200dbfcbed0b6742bbe05a4f1abce532c4fef20918228a651647a3e75d',
    'fraudTest.csv': '12d553ab19440c752d2531ee1af44bb64f12cc3d3839f1649f19e81c230545f0',
}


def checksum(path):
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def historical_date(iso_date):
    year, month, day = iso_date.split('-')
    return f'{day}/{month}/{year}'


def prepare(source, destination):
    if destination.exists():
        raise FileExistsError(f'{destination} already exists. Choose a new output directory.')
    for name, expected in SOURCE_HASHES.items():
        if checksum(source / name) != expected:
            raise ValueError(f'{name} differs from the version 1 input documented in README.md.')

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as staging:
        staged = Path(staging) / 'raw'
        staged.mkdir()
        shutil.copyfile(source / 'fraudTrain.csv', staged / 'fraud_train.csv')
        with (source / 'fraudTest.csv').open(newline='') as src, (staged / 'fraud_test.csv').open('w', newline='') as out:
            reader = csv.DictReader(src)
            writer = csv.DictWriter(out, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                # The historical test export has minute precision and day-first dates.
                # Preserve that representation to replay the original policy.
                date, time = row['trans_date_trans_time'].split(' ')
                row['trans_date_trans_time'] = f'{historical_date(date)} {time[:5]}'
                row['dob'] = historical_date(row['dob'])
                for column in ('amt', 'lat', 'long', 'merch_lat', 'merch_long'):
                    row[column] = format(float(row[column]), '.6f').rstrip('0').rstrip('.')
                writer.writerow(row)
        staged.rename(destination)
    print(f'Prepared training and test files in {destination}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='Folder containing fraudTrain.csv and fraudTest.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'Data' / 'raw')
    args = parser.parse_args()
    prepare(args.source, args.output)
