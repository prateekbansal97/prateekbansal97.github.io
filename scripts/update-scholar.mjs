import 'dotenv/config';
import fetch from 'cross-fetch';
import fs from 'fs';
//import cheerio from 'cheerio';
import { load } from 'cheerio';

const key = process.env.SERPAPI_KEY || '';
const authorId = process.env.SCHOLAR_AUTHOR_ID || 'B0OVVlEAAAAJ';
const OUT = 'src/content/publications.json';

function normalizeAuthors(auth) {
  if (Array.isArray(auth)) {
    return auth
      .map((x) => (typeof x === 'string' ? x : x?.name))
      .filter(Boolean)
      .join(', ');
  }
  if (typeof auth === 'string') return auth;
  if (auth && typeof auth === 'object' && auth.name) return String(auth.name);
  return '';
}

function toIntMaybe(v) {
  const n = parseInt(String(v), 10);
  return Number.isFinite(n) ? n : undefined;
}

function dedupeByTitle(list) {
  const seen = new Set();
  return list.filter((p) => {
    const key = (p.title || '').toLowerCase().trim();
    if (!key || seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

async function fetchFromSerpApi(authorId, apiKey) {
  const url = new URL('https://serpapi.com/search.json');
  url.searchParams.set('engine', 'google_scholar_author');
  url.searchParams.set('author_id', authorId);
  url.searchParams.set('api_key', apiKey);
  url.searchParams.set('sort', 'pubdate'); // newest first

  const res = await fetch(url.toString(), { headers: { 'User-Agent': 'Mozilla/5.0' } });
  if (!res.ok) {
    throw new Error(`SerpAPI HTTP ${res.status}`);
  }
  const data = await res.json();

  const arts = Array.isArray(data.articles) ? data.articles : [];
  const pubs = arts.map((a) => {
    const authors = normalizeAuthors(a.authors);
    const pdfRes =
      Array.isArray(a.resources) &&
      a.resources.find((r) => (r.file_format || '').toString().toLowerCase() === 'pdf');
    const pdf = pdfRes?.link || pdfRes?.url || '';

    return {
      title: a.title || '',
      authors,
      venue: a.publication || a.journal || a.publisher || '',
      year: toIntMaybe(a.year),
      url: a.link || a.result_id || '',
      pdf,
      code: '',
      citation: ''
    };
  });

  return pubs.filter((p) => p.title);
}

async function fetchFromScholarHtml(authorId) {
  const url = `https://scholar.google.com/citations?user=${authorId}&hl=en&view_op=list_works&sortby=pubdate`;
  const res = await fetch(url, {
    headers: {
      // Lightweight UA helps avoid trivial blocking
      'User-Agent':
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119 Safari/537.36'
    }
  });
  if (!res.ok) throw new Error(`Scholar HTML HTTP ${res.status}`);
  const html = await res.text();
  //const $ = cheerio.load(html);
    const $ = load(html);

  const pubs = [];
  $('.gsc_a_tr').each((_, el) => {
    const titleEl = $(el).find('.gsc_a_at').first();
    const title = titleEl.text().trim();
    const href = titleEl.attr('href');
    const url = href ? new URL(href, 'https://scholar.google.com').toString() : '';
    const meta = $(el).find('.gsc_a_t .gs_gray');
    const authors = (meta.eq(0).text() || '').trim();
    const venue = (meta.eq(1).text() || '').trim();
    const year = toIntMaybe($(el).find('.gsc_a_y span').text().trim());

    if (title) {
      pubs.push({
        title,
        authors,
        venue,
        year,
        url,
        pdf: '',
        code: '',
        citation: ''
      });
    }
  });

  return pubs;
}

function sortByYearDesc(pubs) {
  return pubs.sort((a, b) => (b.year || 0) - (a.year || 0));
}

async function main() {
  let pubs = [];
  if (key) {
    try {
      console.log('Fetching publications via SerpAPI…');
      pubs = await fetchFromSerpApi(authorId, key);
    } catch (e) {
      console.warn('SerpAPI failed:', e.message);
    }
  }

  if (!pubs.length) {
    try {
      console.log('Falling back to direct Scholar HTML fetch…');
      pubs = await fetchFromScholarHtml(authorId);
    } catch (e) {
      console.warn('Scholar HTML fetch failed:', e.message);
    }
  }

  pubs = dedupeByTitle(pubs);
  pubs = sortByYearDesc(pubs);

  if (!pubs.length) {
    console.error('No publications found. Check SERPAPI_KEY or network/HTML fetch.');
    process.exit(1);
  }

  fs.mkdirSync('src/content', { recursive: true });
  fs.writeFileSync(OUT, JSON.stringify(pubs, null, 2));
  console.log(`Wrote ${pubs.length} publications to ${OUT}`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
