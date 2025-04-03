import { publications } from './publications-data.js';
import { renderPublications } from './render-publications.js';

document.addEventListener("DOMContentLoaded", () => {
    renderPublications(publications);
});
