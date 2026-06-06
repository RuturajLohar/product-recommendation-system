# Docker: `compose down` fails or hangs (HTTP 500 / daemon errors)

If `docker compose down` prints nothing for a long time, errors with **500 Internal Server Error**, or says the server does not support the API version, the **Docker engine** (Docker Desktop VM) is unhealthy—not this repository.

## Quick fix (try in order)

1. **Quit Docker Desktop fully**  
   Menu bar whale icon → **Quit Docker Desktop**. Wait 10 seconds. Open Docker Desktop again and wait until it says **Docker is running**.

2. **Verify the daemon** (should return in 1–2 seconds):
   ```bash
   docker version
   ```
   You should see both **Client** and **Server** sections. If Server is missing or errors, the engine is still down.

3. **Troubleshoot from Docker Desktop**  
   **Settings** (gear) → **Troubleshoot** → **Restart** (or **Reset to factory defaults** if restarts do not help; this removes local images/volumes—only if you accept data loss).

4. **macOS: ensure no stale socket** (rare)  
   After a crash, only a full Docker Desktop restart fixes the VM.

5. **CLI context**  
   ```bash
   docker context ls
   docker context use desktop-linux
   ```
   Then retry `docker compose down` from `recommender_platform/`.

## After the engine is healthy

From `recommender_platform/`:

```bash
docker compose down
```

Remove named volumes too (Postgres + Qdrant data):

```bash
docker compose down -v
```

Or use the repo helper:

```bash
./scripts/docker-teardown.sh
./scripts/docker-teardown.sh --volumes   # same as down -v
```

## When *nothing* works (`compose down`, `docker ps`, `docker info` hang or error)

The CLI can still show **Client** info while **Server** never appears or returns **HTTP 500**. No repo script can remove containers until the engine responds.

Do this **in order**:

### 1. Force-quit Docker Desktop (not just close the window)

- **Activity Monitor** → search **Docker** → select **Docker Desktop** (and related processes if present, e.g. `com.docker.backend`) → **Force Quit**.
- Wait **30 seconds**, then open **Docker Desktop** from Applications and wait until **Docker is running** (can take 1–2 minutes after a bad state).

### 2. Confirm the daemon (must be fast)

```bash
docker version
```

You need a **Server:** block. If only **Client:** appears or it errors, the engine is still broken — repeat step 1 or go to step 3.

### 3. Docker Desktop built-in reset

Open **Docker Desktop** → **Settings** (gear) → **Troubleshoot**:

- Click **Restart Docker Desktop**.
- If still broken: **Reset to factory defaults** (this **deletes** local images, containers, and volumes — only if you accept that).

### 4. Reboot macOS

A full restart clears stuck VM networking and file locks that sometimes survive a force-quit.

### 5. Reinstall Docker Desktop

Download the latest **Docker Desktop for Mac (Apple Silicon or Intel)** from Docker’s site, replace the app, and install. This fixes corrupted VM disks.

### 6. Optional: use another engine (advanced)

If you need containers **today** and Desktop keeps failing, some developers use **Colima** or **Rancher Desktop** as the Docker-compatible runtime, then point the CLI with `docker context`. That is a separate setup from this project.

---

## Why this happens

Docker Desktop runs a Linux VM. Updates, sleep/wake, or resource pressure can leave the API in a bad state until you restart the app. A **500** from `docker.sock` almost always means **restart Docker Desktop**, not a bug in `docker-compose.yml`.
