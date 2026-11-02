# Deployment guide

The React/Vite frontend and the Python API are separate deployables. The backend owns the camera stream, model files, inspection worker, and ESP connection; Netlify hosts only the static frontend.

## Before running the backend

1. If `config.yaml` does not already exist, copy `config.example.yaml` to it; otherwise edit the existing file. Enter local camera/ESP settings, and never commit secrets.
2. Put trained model files in `models/` using the paths configured in `config.yaml`.
3. Verify the machine running the backend can reach the RTSP camera and ESP over the required network.

Model weights and datasets are intentionally not included in Docker build context or deployment branches. The repository's tracked LFS pointer is not a substitute for a trained local checkpoint; verify files before starting inference.

## Run backend directly

From the repository root, create a Python environment and install dependencies:

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

Start the CORS-enabled API:

```bash
uvicorn backend_asgi:app --host 0.0.0.0 --port 5004
```

The API is available at `http://localhost:5004`. The Vite development server proxies API requests to that address by default.

## Run backend with Docker Compose

Docker is CPU-only by default; GPU passthrough requires a separate NVIDIA/CUDA image matched to the host driver and PyTorch version.

1. Create the private config and provide weights:

   ```powershell
   if (!(Test-Path config.yaml)) { Copy-Item config.example.yaml config.yaml }
   New-Item -ItemType Directory -Force models
   if (!(Test-Path .env)) { Copy-Item .env.example .env }
   ```

   Then edit `config.yaml`, place the model files under `models/`, and set the correct local values in `.env`.

   macOS/Linux equivalent: `cp -n config.example.yaml config.yaml` and `cp -n .env.example .env`.

2. Build and start:

   ```bash
   docker compose up --build
   ```

3. Check `http://localhost:5004/status` and view logs with `docker compose logs -f backend`.
4. Stop with `docker compose down`.

`docker-compose.yml` mounts `config.yaml` and `models/` at runtime, so neither is baked into the image. The API port is bound to host loopback (`127.0.0.1`) by default, suitable for a local reverse tunnel. The backend container still needs a network route to the RTSP camera and ESP. Test those routes from inside the Docker host before relying on the container. On Windows, camera/ESP reachability also depends on Docker Desktop networking and firewall rules.

## Deploy frontend to Netlify

The checked-in `netlify.toml` configures the `frontend/` base directory, `npm run build`, and `dist/` publish directory.

1. Connect the repository to Netlify; set the site environment variable `VITE_API_BASE_URL` to the HTTPS origin of the backend, for example `https://api.example.com`.
2. Set `FRONTEND_ORIGINS` on the backend to the exact Netlify site origin, e.g. `https://your-site.netlify.app` (add the custom domain if used).
3. Start the API on a persistent backend host with:

   ```bash
   uvicorn backend_asgi:app --host 0.0.0.0 --port ${PORT:-5004}
   ```

4. Redeploy the Netlify site after changing `VITE_API_BASE_URL`.

Netlify cannot directly reach private addresses such as `192.168.x.x`. If the camera and ESP remain on a home/office LAN, keep the backend on that LAN and provide a secured VPN/reverse-proxy route for the frontend, or move the devices/backend to a network with a safe route. **Do not expose RTSP, ESP TCP control, or this unauthenticated API directly to the public Internet.** Add authentication and firewall/IP restrictions before public exposure; CORS is not authentication.

Never put RTSP credentials, model weights, ESP credentials, or private IPs in `VITE_*` variables—the frontend bundle is public. Keep them on the backend.

## Connect a LAN backend to Netlify with Tailscale Funnel (optional)

Funnel can publish the local HTTP backend through an HTTPS URL while the backend stays on the PC that can reach the camera and ESP. Netlify's browser frontend calls that URL; Funnel does not make the private camera or ESP address public. Visitors do not need to install Tailscale, so the Funnel URL itself is public.

**Security gate:** the current FastAPI API has no authentication and includes inspection controls and a live video stream. Do not enable public Funnel until authentication/authorization is added or a protecting access proxy is in place. CORS only controls browser origins; it does not stop direct HTTP clients. Never funnel RTSP or ESP TCP port `12345`.

1. Install/sign in to Tailscale on the backend PC. Enable MagicDNS and HTTPS certificates, and approve the Funnel node attribute when prompted. Funnel requires a recent Tailscale client and listens publicly on HTTPS ports `443`, `8443`, or `10000`.
2. Run the API on loopback so it is not directly exposed on the LAN:

   ```bash
   uvicorn backend_asgi:app --host 127.0.0.1 --port 5004
   ```

   If using Docker Compose, the default `127.0.0.1:5004:5004` binding already makes the API available to Funnel on the host.

3. In another terminal, publish the local HTTP service:

   ```bash
   tailscale funnel --bg 5004
   tailscale funnel status
   ```

   Use the HTTPS `*.ts.net` URL shown by the status command. Tailscale documents Funnel as available on its plans, but it is still a beta feature with non-configurable bandwidth limits; the MJPEG `/video` stream can consume substantial bandwidth.

4. Set Netlify `VITE_API_BASE_URL` to that HTTPS Funnel origin. Set backend `FRONTEND_ORIGINS` to the exact Netlify site origin, for example `https://your-site.netlify.app`, then restart the backend and rebuild/redeploy the frontend.
5. Disable Funnel when no longer needed with `tailscale funnel reset` (this resets Funnel configuration on that device).

Official references: [Tailscale Funnel](https://tailscale.com/docs/features/tailscale-funnel) · [Funnel CLI](https://tailscale.com/docs/reference/tailscale-cli/funnel).

Security follow-up: the repository's original history may already contain an older tracked camera URL with credentials. Rotate that camera password; removing a file in a later commit does not erase it from Git history.

## Current limitation

The image acquisition/ROI logic is expected to change. Keep model preprocessing behind the algorithm adapter and revisit the input contract in `docs/ANOMALY_MODELS_INTEGRATION.md` before deploying DRAEM or SK-RD4AD.
