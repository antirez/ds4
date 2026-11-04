import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'DwarfStar',
  description: 'Run excellent large language models on consumer hardware — DeepSeek V4 Flash, GLM 5.x, Qwen3.8 Flash Next on Metal, CUDA, and ROCm.',
  lang: 'en-US',
  base: '/ds4/',

  head: [
    ['link', { rel: 'icon', href: '/ds4/logo.svg', type: 'image/svg+xml' }],
    ['meta', { property: 'og:image', content: '/ds4/logo.svg' }],
    ['meta', { name: 'theme-color', content: '#2563eb' }],
  ],

  themeConfig: {
    logo: '/logo.svg',

    nav: [
      { text: 'Home', link: '/' },
      { text: 'Getting Started', link: '/getting-started' },
      { text: 'Models', link: '/models' },
      { text: 'Serving', link: '/server' },
      { text: 'GitHub', link: 'https://github.com/antirez/ds4' },
    ],

    sidebar: [
      {
        text: 'Getting Started',
        items: [
          { text: 'Quick Start', link: '/getting-started' },
          { text: 'Models & Vision', link: '/models' },
        ],
      },
      {
        text: 'Platform Guides',
        items: [
          { text: 'Apple Silicon (Metal)', link: '/metal' },
          { text: 'DGX Spark', link: '/dgx-spark' },
          { text: 'Strix Halo / ROCm', link: '/strix-halo' },
          { text: 'CUDA Multi-GPU', link: '/cuda-multi-gpu' },
        ],
      },
      {
        text: 'Usage',
        items: [
          { text: 'Serving', link: '/server' },
          { text: 'Clients', link: '/clients' },
          { text: 'Performance', link: '/performance' },
          { text: 'SSD Streaming', link: '/ssd-streaming' },
        ],
      },
      {
        text: 'Advanced',
        items: [
          { text: 'Distributed Inference', link: '/distributed' },
          { text: 'Speculative Decoding', link: '/speculative-decoding' },
          { text: 'Qwen3.8 Flash Next', link: '/qwen38-flash-next' },
        ],
      },
      {
        text: 'Development',
        items: [
          { text: 'Testing', link: '/testing' },
          { text: 'Contributing', link: '/contributing' },
        ],
      },
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/antirez/ds4' },
    ],

    editLink: {
      pattern: 'https://github.com/antirez/ds4/edit/main/site/:path',
      text: 'Edit this page on GitHub',
    },

    lastUpdated: true,
    search: { provider: 'local' },
    footer: {
      message: 'Built with DwarfStar. Powered by VitePress.',
    },
  },
})