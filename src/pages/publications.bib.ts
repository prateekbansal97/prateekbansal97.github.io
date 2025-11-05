import pubs from '@/content/publications.json';

export async function GET() {
  const lines = (pubs as any[]).map((p, i) => {
    const key = (p.title || `pub${i}`)
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '_')
      .replace(/^_+|_+$/g, '');
    const year = p.year || '';
    const authors = (p.authors || '').replace(/&/g, 'and');
    const venue = p.venue || '';
    const url = p.url || '';
    return `@article{${key}_${year},
  title = {${p.title || ''}},
  author = {${authors}},
  journal = {${venue}},
  year = {${year}},
  url = {${url}}
}`;
  });
  return new Response(lines.join('\n\n'), {
    headers: { 'Content-Type': 'application/x-bibtex; charset=utf-8' }
  });
}
