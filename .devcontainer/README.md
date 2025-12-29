# Dev Container Setup for MIOpen

This directory contains the configuration for VS Code Dev Containers extension, configured to match your Docker command:

```bash
docker run -it -v $HOME:/data --workdir /data --privileged --rm \
  --device=/dev/kfd --device /dev/dri:/dev/dri:rw \
  --volume /dev/dri:/dev/dri:rw -v /var/lib/docker/:/var/lib/docker \
  --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  rocm/miopen:ci_bbedc1
```

## Quick Start (with Remote - SSH)

Since you're using Remote - SSH extension:

1. **Connect via Remote - SSH first**:
   - Press `F1` (or `Ctrl+Shift+P` / `Cmd+Shift+P`)
   - Type "Remote-SSH: Connect to Host"
   - Select your remote host
   - Open the `rocm-libraries` folder on the remote machine

2. **Then reopen in Container**:
   - Press `F1` again
   - Type "Dev Containers: Reopen in Container"
   - Select it
   - VS Code will build/start the container on the remote machine with all the required settings

**Note**: The Dev Containers extension works on top of Remote - SSH. It will:
- Use Docker on the remote machine (where you SSH to)
- Create/start the container on that remote machine
- Mount your workspace from the remote filesystem into the container

## Configuration Details

The `devcontainer.json` is configured to:
- Use image: `rocm/miopen:ci_bbedc1`
- Mount workspace to `/data` (matching your `-v $HOME:/data`)
- Enable GPU access (`/dev/kfd`, `/dev/dri`)
- Set privileged mode and required capabilities
- Set working directory to `/data`

## Workspace Structure

- **Workspace folder in container**: `/data`
- **Build directory**: `/data/build/bin/` (where test executables are located)
- **Source code**: `/data/projects/miopen/` (mounted from your local workspace)

## Debugging

Once connected to the container:
- Use the launch.json configurations in `.vscode/launch.json`
- Set breakpoints in your code
- Press `F5` to start debugging

## Updating the Image Tag

If you need to use a different image tag (e.g., `ci_xxxxx`), update line 2 in `devcontainer.json`:
```json
"image": "rocm/miopen:ci_xxxxx",
```

## Prerequisites

Since you're using Remote - SSH:
- **Docker must be installed and running** on the remote machine (where you SSH to)
- Your SSH user must have permissions to run Docker (usually requires being in the `docker` group)
- The Docker image `rocm/miopen:ci_bbedc1` must be available on the remote machine (or Docker will pull it)

## Troubleshooting

- If the container doesn't start, check Docker is running on the remote machine: `ssh <host> docker ps`
- Check the VS Code Output panel (View → Output → Dev Containers) for errors
- Verify your Docker image exists on the remote machine: `ssh <host> docker images | grep rocm/miopen`
- If GPU access doesn't work, ensure Docker on the remote machine has access to `/dev/kfd` and `/dev/dri`
- The container runs with `--privileged` mode, which may require appropriate permissions on the remote machine
- If you get permission errors, ensure your SSH user is in the `docker` group: `groups` (should include "docker")

