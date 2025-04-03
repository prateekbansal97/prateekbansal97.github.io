export function renderPublications(publications, containerId = 'publications-grid') {
    const container = document.getElementById(containerId);

    if (!container) {
        console.warn(`Container #${containerId} not found.`);
        return;
    }

    if (!Array.isArray(publications) || publications.length === 0) {
        container.innerHTML = `<p>No publications available at the moment.</p>`;
        return;
    }

    publications.forEach((pub, index) => {
        const pubDiv = document.createElement('div');
        pubDiv.className = 'publication';

        const authorsFormatted = pub.authors.replace("Bansal, P.", "<b>Bansal, P.</b>");
        const journalFormatted = `<i>${pub.journal}</i>`;
        const categoryLabel = pub.category.toUpperCase();

        pubDiv.innerHTML = `
            <p class="author-info">${index + 1}. ${authorsFormatted} <br><strong>${pub.title}</strong>, ${journalFormatted}, ${pub.year} <a class="publication-link" href="${pub.link}" target="_blank">Link</a></p>
            <span class="category ${pub.category}">${categoryLabel}</span>
            <img src="${pub.image}" alt="Figure from the publication">
        `;

        container.appendChild(pubDiv);
    });
}
