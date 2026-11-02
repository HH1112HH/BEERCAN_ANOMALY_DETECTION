# React/Vite dashboard

The frontend is deployed independently from the Python API.

```bash
npm ci
npm run dev
npm run build
```

Development proxy target is `VITE_DEV_API_TARGET` (default `http://127.0.0.1:5004`). For a static deployment, set `VITE_API_BASE_URL` to the public API origin before building. See the repository root `README.md` and `DESIGN.md` for deployment and design handoff details.
