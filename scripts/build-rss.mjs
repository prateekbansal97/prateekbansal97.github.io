import { writeFileSync, readFileSync } from 'fs';
import { rss } from '@astrojs/rss';

const site = 'https://your-domain.com';
const pubs = JSON.parse(readFileSync('src/content/publications.json', 'utf8'));

const xml = rss({
  title: 'Publications — Prateek Bansal',
  description: 'Latest publications and preprints',
  site,
  items: pubs.slice(0, 30).map((p) => ({
    title: p.title,
    pubDate: new Date(`${p.year || '2000'}-01-01`),
    link: p.url || site,
    description: `${p.authors} (${p.year}). ${p.venue}.`
  }))
}).body;

writeFileSync('dist/rss.xml', xml);
console.log('Built RSS feed at dist/rss.xml');
