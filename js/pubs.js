const publications = [
    {
        title: "Understanding Supramolecular Assembly of Supercharged Proteins",
        image: "img/WebsiteImage_Supercharged.jpg",
        authors: "Jacobs, M., Bansal, P. , Shukla, D. , Schroeder, C.",
        journal: "ACS Central Science",
        year: "2022",
        category: "md",
        link: "https://pubs.acs.org/doi/full/10.1021/acscentsci.2c00730"
    },
    {
        title: "Activation Mechanism of the Human Smoothened Receptor",
        image: "img/WebsiteImage_SMOActivation.png",
        authors: "Bansal, P., Dutta, S. , Shukla, D.",
        year: "2023",
        journal: "Biophysical Journal",
        category: "md",
        link: "https://www.sciencedirect.com/science/article/pii/S0006349523001601"
    },
    // {
    //     title: "Title of Paper 3",
    //     image: "path/to/image3.jpg",
    //     authors: "Your Name, Co-author Name, etc.",
    //     journal: "Journal Name",
    //     year: "2023",
    //     category: "experiment",
    //     link: "link_to_paper_3"
    // }
];

const publicationsGrid = document.getElementById('publications-grid');

// publications.forEach((pub, index) => {
//     const publicationDiv = document.createElement('div');
//     publicationDiv.className = 'publication';

//     // Bold "Bansal, P." and italicize the journal name
//     const authorsFormatted = pub.authors.replace("Bansal, P.", "<b>Bansal, P.</b>");
//     const journalFormatted = `<i>${pub.journal}</i>`;

//     publicationDiv.innerHTML = `
//         <h4>${index + 1}. ${pub.title}</h4>
//         <img src="${pub.image}" alt="Image representing the paper">
//         <p>${authorsFormatted}</p>
//         <p>${journalFormatted}, ${pub.year}</p>
//         <span class="category ${pub.category}">${pub.category.toUpperCase()}</span>
//         <a href="${pub.link}">Read more</a>
//     `;
//     publicationsGrid.appendChild(publicationDiv);
// });

publications.forEach((pub, index) => {
    const publicationDiv = document.createElement('div');
    publicationDiv.className = 'publication';
    const authorsFormatted = pub.authors.replace("Bansal, P.", "<b>Bansal, P.</b>");
    const journalFormatted = `<i>${pub.journal}</i>`;

    publicationDiv.innerHTML = `
        <p class ="author-info">${index + 1}. ${authorsFormatted} ${pub.title}, ${journalFormatted}, ${pub.year} <a href="${pub.link}">Link</a></p>
        <span class="category ${pub.category}">${pub.category.toUpperCase()}</span>
        <img src="${pub.image}" alt="Image representing the paper">
        
    `;
    publicationsGrid.appendChild(publicationDiv);
});