import { defineConfig } from 'astro/config';
import tailwind from '@astrojs/tailwind';
import sitemap from '@astrojs/sitemap';

export default defineConfig({
  site: 'https://prateekbansal97.github.io', // update for canonical
  integrations: [
    tailwind({
      applyBaseStyles: false // we handle our own base for academic typography
    }),
    sitemap()
  ],
  prefetch: true
});
