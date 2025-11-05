import pubs from '@/content/publications.json';

export const prerender = true;

function escapeXml(s: string) {
  return (s || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function rfc1123(d: Date) {
  return d.toUTCString();
}

function buildRssXml(site: string, items: any[]) {
  const channelTitle = 'Publications — Prateek Bansal';
  const channelDesc = 'Latest publications and preprints';
  const now = rfc1123(new Date());

  const itemsXml = items.slice(0, 30).map((p) => {
    const link = p.url || site;
    const pubDate = rfc1123(new Date(`${p.year || '2000'}-01-01`));
    const desc = `${p.authors || ''} (${p.year || ''}). ${p.venue || ''}.`;
    return `<item>
  <title>${escapeXml(p.title || '')}</title>
  <link>${escapeXml(link)}</link>
  <guid>${escapeXml(link)}</guid>
  <pubDate>${pubDate}</pubDate>
  <description>${escapeXml(desc)}</description>
</item>`;
  }).join('\n');

  return `<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
<channel>
  <title>${escapeXml(channelTitle)}</title>
  <link>${escapeXml(site)}</link>
  <description>${escapeXml(channelDesc)}</description>
  <lastBuildDate>${now}</lastBuildDate>
${itemsXml}
</channel>
</rss>`;
}

export async function GET() {
  const site = import.meta.env.SITE_URL || 'https://prateekbansal97.github.io';
  const xml = buildRssXml(site, pubs as any[]);
  return new Response(xml, {
    headers: { 'Content-Type': 'application/rss+xml; charset=utf-8' }
  });
}
