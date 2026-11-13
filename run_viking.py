#!/usr/bin/env python3
"""Execute the Viking notebook headlessly inside the NeuroStrike white_box directory.

All notebook outputs (including partial outputs if a cell fails) are saved.
Training checkpoints are saved by A3b before the long StrongREJECT evaluation.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--notebook',type=Path,required=True)
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--executed',type=Path,required=True)
    args = parser.parse_args()
    repo=args.repo.resolve()
    notebook=args.notebook.resolve()
    out=args.executed.resolve()
    assert (repo/'white_box'/'util_model.py').is_file(), f'NeuroStrike repo missing: {repo}'
    assert notebook.is_file(), f'Notebook missing: {notebook}'
    out.parent.mkdir(parents=True,exist_ok=True)
    os.environ['NEUROSTRIKE_REPO']=str(repo)
    os.environ.setdefault('SUBNETWORK_RESULTS_DIR',str(out.parent))
    nb=nbformat.read(notebook,as_version=4)
    client=NotebookClient(nb, timeout=None, kernel_name='neurostrike-viking',
                          resources={'metadata':{'path':str(repo/'white_box')}},
                          allow_errors=False)
    print(f'Running {notebook.name} in {repo/"white_box"}',flush=True)
    print(f'Results: {os.environ["SUBNETWORK_RESULTS_DIR"]}',flush=True)
    start=time.monotonic()
    try:
        client.execute()
        print('Notebook completed successfully.',flush=True)
    finally:
        # Preserve executed cells and exception outputs even if a later cell fails.
        nbformat.write(nb,out)
        print(f'Executed notebook saved: {out}; elapsed {time.monotonic()-start:.0f}s',flush=True)


if __name__=='__main__':
    main()
