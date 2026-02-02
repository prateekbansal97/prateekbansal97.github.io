import { defineConfig } from 'astro/config';
import tailwind from '@astrojs/tailwind';
import sitemap from '@astrojs/sitemap';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

export default defineConfig({
  site: 'https://prateekbansal97.github.io', // update for canonical
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
    shikiConfig: {
      theme: 'dracula',
    },
  },
  integrations: [
    tailwind({
      applyBaseStyles: false // we handle our own base for academic typography
    }),
    sitemap()
  ],
  prefetch: true
});
