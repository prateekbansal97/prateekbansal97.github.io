import { writeFileSync, readFileSync } from 'fs';
import rss from '@astrojs/rss'; // ✅ default import

const site = process.env.SITE_URL || 'https://prateekbansal97.github.io';
const pubs = JSON.parse(readFileSync('src/content/publications.json', 'utf8'));

async function getXmlFromRss(result) {
  // Some versions return a Response; some may return a string.
  if (typeof result === 'string') return result;

  if (result && typeof result.text === 'function') {
    // Standard Web Response (undici)
    return await result.text();
  }

  // Fallback: stream body (rare)
  if (result && result.body) {
    const chunks = [];
    for await (const chunk of result.body) chunks.push(Buffer.from(chunk));
    return Buffer.concat(chunks).toString('utf8');
  }

  throw new Error('Unexpected return from @astrojs/rss');
}

const res = rss({
  title: 'Publications — Prateek Bansal',
  description: 'Latest publications and preprints',
  site,
  items: pubs.slice(0, 30).map((p) => ({
    title: p.title,
    pubDate: new Date(`${p.year || '2000'}-01-01`),
    link: p.url || site,
    description: `${p.authors} (${p.year}). ${p.venue}.`
  }))
});

const xml = await getXmlFromRss(res);
writeFileSync('dist/rss.xml', xml, 'utf8');
console.log('Built RSS feed at dist/rss.xml');
