# Deployment guide

The React/Vite frontend and the Python API are separate deployables. The backend owns the camera stream, model files, inspection worker, and ESP connection; Netlify hosts only the static frontend.

## Before running the backend

1. Copy `config.example.yaml` to `config.yaml` and enter the local camera/ESP settings. Do not commit the real `config.yaml` or secrets.
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
   Copy-Item config.example.yaml config.yaml
   New-Item -ItemType Directory -Force models
   Copy-Item .env.example .env
   ```

   Then edit `config.yaml`, place the model files under `models/`, and set the correct local values in `.env`.

2. Build and start:

   ```bash
   docker compose up --build
   ```

3. Check `http://localhost:5004/status` and view logs with `docker compose logs -f backend`.
4. Stop with `docker compose down`.

`docker-compose.yml` mounts `config.yaml` and `models/` at runtime, so neither is baked into the image. The backend container still needs a network route to the RTSP camera and ESP. Test those routes from inside the Docker host before relying on the container. On Windows, camera/ESP reachability also depends on Docker Desktop networking and firewall rules.

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

## Current limitation

The image acquisition/ROI logic is expected to change. Keep model preprocessing behind the algorithm adapter and revisit the input contract in `docs/ANOMALY_MODELS_INTEGRATION.md` before deploying DRAEM or SK-RD4AD.
