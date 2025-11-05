import 'dotenv/config';
import fetch from 'cross-fetch';
import fs from 'fs';

const token = process.env.GITHUB_TOKEN;
if (!token) {
  console.error('Missing GITHUB_TOKEN in .env');
  process.exit(1);
}

const query = `
  query($login:String!) {
    user(login:$login) {
      pinnedItems(first:6, types: REPOSITORY) {
        nodes {
          ... on Repository {
            name
            description
            url
            stargazerCount
            forkCount
            primaryLanguage { name color }
            repositoryTopics(first: 8) { nodes { topic { name } } }
          }
        }
      }
    }
  }
`;

async function main() {
  const res = await fetch('https://api.github.com/graphql', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Authorization': `bearer ${token}` },
    body: JSON.stringify({ query, variables: { login: 'prateekbansal97' } })
  });
  const json = await res.json();
  if (json.errors) {
    console.error(json.errors);
    process.exit(1);
  }
  const nodes = json.data.user.pinnedItems.nodes ?? [];
  const mapped = nodes.map((r) => ({
    name: r.name,
    description: r.description,
    url: r.url,
    stargazerCount: r.stargazerCount,
    forkCount: r.forkCount,
    primaryLanguage: r.primaryLanguage,
    topics: (r.repositoryTopics?.nodes ?? []).map((n) => n.topic.name)
  }));
  fs.writeFileSync('src/content/projects.json', JSON.stringify(mapped, null, 2));
  console.log(`Wrote ${mapped.length} projects to src/content/projects.json`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
