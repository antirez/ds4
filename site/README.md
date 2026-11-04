# DwarfStar Documentation Site

Documentation website for DwarfStar, built with [VitePress](https://vitepress.dev).

## Development

```sh
cd site
npm install
npm run dev
```

Open `http://localhost:4321/ds4/`.

## Build

```sh
npm run build      # outputs to .vitepress/dist
npm run preview    # preview the production build
```

## Structure

```
site/
├── .vitepress/
│   ├── config.mjs          # Site config: nav, sidebar, search
│   └── theme/
│       ├── index.js        # Theme entry
│       └── custom.css      # DwarfStar styling
├── public/                 # Static assets (logo, benchmark charts)
├── index.md                # Landing page
├── getting-started.md
├── models.md
├── metal.md
├── dgx-spark.md
├── strix-halo.md
├── cuda-multi-gpu.md
├── server.md
├── clients.md
├── performance.md
├── ssd-streaming.md
├── distributed.md
├── speculative-decoding.md
├── qwen38-flash-next.md
├── testing.md
└── contributing.md
```

## Deployment

Deployed automatically to GitHub Pages via `.github/workflows/deploy-pages.yml`
on pushes to `main` that touch `site/`.

Published at: `https://antirez.github.io/ds4/`

### Enabling Pages

In the repository settings, set **Pages → Build and deployment → Source** to
**GitHub Actions**.

## Third-party logos

The platform cards on the landing page use Apple, NVIDIA, AMD, and DeepSeek
logos from [Simple Icons](https://simpleicons.org) (CC0 1.0). They are inlined
as `currentColor` SVG in `index.md` so they follow the light and dark theme.
Logos are used to identify the hardware and models DwarfStar supports; all
trademarks belong to their respective owners.
