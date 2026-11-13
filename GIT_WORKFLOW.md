# Local Git workflow

The repository uses branch `main`. Git does not upload files unless you explicitly
configure a remote and push. No remote is configured by this setup.

Set your commit identity locally before the first commit:

```bash
git config user.name "Your Name"
git config user.email "your-email@example.com"
git status
git diff --cached --stat
git commit -m "Record NeuroStrike experiment code and documentation"
```

After changes, review `git diff`, stage the files you want, and commit.
Credentials, environments, caches, the upstream NeuroStrike checkout, logs,
checkpoints and results are ignored. These files remain on disk. Root notebooks
are staged as source; both had no saved cell outputs at setup. Inspect future
outputs before committing or publishing.
Experiment results are not backed up by this Git repository. To track a selected
non-sensitive report or small CSV, review it and use `git add -f path/to/file`.
No Git LFS or remote service is required for this local setup.

## Optional remote, later

Create an empty private repository in your chosen hosting service, then connect it:

```bash
git remote add origin YOUR_REPOSITORY_URL
git push -u origin main
```

Do this after making the first local commit. No remote was created or configured.
The separately managed `NeuroStrike/` directory is ignored, not a submodule.
