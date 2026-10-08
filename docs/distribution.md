# Additional distribution channels

The Python package on PyPI remains the inference implementation. Additional
packages provide launchers, an HTTP SDK, or a serving container. Model weights
remain on Hugging Face and are downloaded when the server starts.

## Release status

Release status for 0.3.0. Registry publication and installation checks are
separate from adding package source files to this repository.

| Channel | Implementation in this repository | Status / remaining release step |
|---|---|---|
| PyPI | Existing Python package, 0.3.0 | Existing release |
| npm | `packages/npm`: CLI + typed ESM HTTP SDK | Published as [`sokudan@0.3.0`](https://www.npmjs.com/package/sokudan); clean install verified; OIDC setup remains |
| Homebrew own tap | `Formula/sokudan.rb`: uv launcher and checksummed Python source | Available on main; macOS install, `brew test`, and CLI startup passed CI |
| GHCR | `Dockerfile.serve`, `container.yml`: Linux amd64 CPU server | Published as [`0.3.0-cpu`](https://github.com/hiroki-abe-58/sokudan/pkgs/container/sokudan); model inference passed; anonymous manifest/config download verified |
| Docker Hub | Same tested serving image | Choose namespace; authenticate and push |
| Conda / conda-forge | Not implemented | Resolve MLX/PyTorch dependencies; build/test recipe; submit |
| Scoop own bucket | `bucket/sokudan.json`: Windows x64 uv launcher | Windows installation, startup, and uninstall are checked by `scoop.yml`; available after merge |
| WinGet / Chocolatey | Not implemented | WinGet needs a supported installer/portable artifact; Chocolatey needs an account and package review |
| Nix / AUR | Not implemented | Resolve native dependencies and test builds on target systems |
| JSR | SDK source is separate from Node-only CLI | Choose JSR scope, validate and publish SDK |

Official repository acceptance (Homebrew core, conda-forge, WinGet, Nixpkgs) is
separate from publishing to a maintainer-owned tap/channel. No submissions to
those repositories are made by these workflows.

## npm

Requirements: Node.js 22+, npm, Python 3 for launcher tests. Runtime consumers of
the SDK do not need Python. CLI consumers need uv, or `SOKUDAN_PYTHON` pointing
to an environment with sokudan and the appropriate extras already installed.

```sh
cd packages/npm
npm ci
npm run typecheck
npm test
npm pack
```

Install `sokudan-0.3.0.tgz` in a separate project to test the actual archive.
The build copies the root LICENSE and NOTICE and generates JS/declarations.
There is no postinstall hook and no npm runtime dependency.

For an HTTP integration check against the real Python server with a fake model:

```sh
# Use an environment with sokudan[serve] and pytest installed.
SOKUDAN_TEST_PYTHON=/path/to/venv/bin/python node scripts/check-python-server.mjs
```

Version 0.3.0 is published under the npm account `genelab`. Install the SDK with
`npm install sokudan`, or the CLI with `npm install -g sokudan` (Node.js 22+).

For a manual release (from `packages/npm`, after verifying the account with
`npm whoami`):

```sh
npm login
npm publish --access public
```

The npm version pins the Python version used by the default CLI, so a matching
non-yanked release must already exist on PyPI. `scripts/check-release.mjs`
checks the tag and Python release before CI publishing.

For subsequent releases, configure an npm trusted publisher:

- GitHub owner: `hiroki-abe-58`
- Repository: `sokudan`
- Workflow filename: `npm.yml`
- GitHub environment: `npm` (create it; required reviewers are optional)

Use npm 11.5.1+ for trusted publishing (workflow uses npm 11 with Node 24).
Push `npm-v<version>` only after the package version and matching PyPI release
are ready. The workflow tests Node 22/24 on Linux, macOS and Windows, then
publishes with OIDC and provenance. It needs no stored npm token. An npm version
cannot be reused; SDK/launcher fixes also need a new matching Python release
under the current shared-version policy.

## Homebrew: third-party tap

Install from the third-party tap:

```sh
brew tap hiroki-abe-58/sokudan https://github.com/hiroki-abe-58/sokudan
brew install hiroki-abe-58/sokudan/sokudan
sokudan serve --port 8000
```

Using this repository as a custom-URL tap avoids creating a second repository.
This is a **launcher formula**, not a Homebrew-core-style bundled Python runtime:
brew installs uv, Python 3.13 and the checksummed PyPI source. On first use uv
builds that source with the `serve` extra in its user cache. Research probes add
`train,bench,torch`. Transitive Python dependencies are resolved at first use,
not locked by the Formula. First use therefore needs network access.
Homebrew removes its files on uninstall; uv and Hugging Face user caches remain.

`homebrew.yml` checks Formula installation, `brew test`, and the first invocation
of `sokudan serve --help` on macOS. This does not claim that every Homebrew/Linux
configuration or GPU backend is tested. Do not submit this launcher to
Homebrew core as if it were an offline, fully bundled Python formula.

On each Python release update the Formula's URL and SHA-256 from PyPI, then rerun
the installation check. Keep its version aligned with the npm release.

## Windows: Scoop third-party bucket

With [Scoop](https://scoop.sh/) already installed, run in PowerShell:

```powershell
scoop bucket add sokudan https://github.com/hiroki-abe-58/sokudan
scoop install sokudan/sokudan
sokudan serve --port 8000
```

The bucket targets Windows x64. It installs the checksummed PyPI source and a
PowerShell launcher, with `main/uv` as its dependency. First use installs Python
3.11 and the `serve` dependencies into uv's cache; `probe-position` additionally
requests `train,bench,torch`. Model weights download when the server starts.
This does not bundle a standalone executable or promise GPU inference support.

`scoop uninstall sokudan` removes the launcher and package source. uv's cache and
Hugging Face model cache remain. Each release must update the manifest version,
source URL, SHA-256 and extracted directory together, and pass Windows CI.

## Serving container: GHCR and Docker Hub

The root `Dockerfile` is still the clean-install regression check.
`Dockerfile.serve` is the non-root CPU serving image. It initially targets
**Linux amd64**; Apple silicon can emulate it but will not use MLX/MPS inside it.
Native arm64 and CUDA images need their own build/runtime validation.

Run the published image:

```sh
docker run --rm --platform linux/amd64 -p 127.0.0.1:8000:8000 \
  -v sokudan-cache:/home/sokudan/.cache/huggingface \
  ghcr.io/hiroki-abe-58/sokudan:0.3.0-cpu
```

The 0.3.0-cpu release digest is
`sha256:c252e252d84cddb1599784e25cff9a6d2bb5e0d77fdc518922f8aa6a23b09836`.
The [publication run](https://github.com/hiroki-abe-58/sokudan/actions/runs/37775222339)
verified model loading and prediction before pushing the image.

To build locally:

```sh
docker build --platform linux/amd64 -f Dockerfile.serve -t sokudan-serve:0.3.0 .
docker run --rm -p 127.0.0.1:8000:8000 \
  -v sokudan-cache:/home/sokudan/.cache/huggingface sokudan-serve:0.3.0
```

The container binds to all interfaces internally; the example publishes only to
host loopback. Inference starts after the first model download. Cache that
download with the named volume. The server does not authenticate clients.

`container.yml` builds and tests on PRs and manual runs. It checks the packaged
CLI, `pip check`, actual model loading, `/health`, and an inference request.
Its manual `publish` option works only on `main` and pushes the **same tested
image** to `ghcr.io/hiroki-abe-58/sokudan:<version>-cpu` and `sha-<commit>`.
After the first push, confirm the GHCR package is public before documenting it
as an anonymous download. Neither a PR nor a tag automatically publishes it.

Docker Hub can receive the same tested image. Once the namespace has been chosen:

```sh
docker login
docker tag sokudan-serve:0.3.0 YOUR_NAMESPACE/sokudan:0.3.0-cpu
docker push YOUR_NAMESPACE/sokudan:0.3.0-cpu
```

For automation, store the Docker Hub username and a scoped write token in GitHub
Actions secrets and add a login/push step after the existing model test. Do not
commit credentials. Dependencies and the model's default Hub revision are not
fully frozen; the image tag describes the packaged code, not immutable model weights.

## Sources

- [npm trusted publishing](https://docs.npmjs.com/trusted-publishers/)
- [uv tools](https://docs.astral.sh/uv/guides/tools/)
- [Homebrew taps](https://docs.brew.sh/How-to-Create-and-Maintain-a-Tap)
- [Homebrew language-specific formulae](https://docs.brew.sh/Language-Specific-Formulae)
- [Scoop manifests](https://github.com/ScoopInstaller/Scoop/wiki/App-Manifests)
- [Scoop buckets](https://github.com/ScoopInstaller/Scoop/wiki/Buckets)
- [GHCR](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry)
- [conda-forge submissions](https://conda-forge.org/docs/maintainer/adding_pkgs/)
- [WinGet submissions](https://learn.microsoft.com/en-us/windows/package-manager/package/)
- [JSR publishing](https://jsr.io/docs/publishing-packages)
