//// Function to toggle dark mode
//function toggleDarkMode() {
//    // Toggle the 'dark-mode' class on the body element
//    document.body.classList.toggle('dark-mode');
//    document.querySelector('nav ul').classList.toggle('dark-mode');
//    
//    // Save the preference to local storage
//    if (document.body.classList.contains('dark-mode')) {
//        localStorage.setItem('theme', 'dark');
//    } else {
//        localStorage.setItem('theme', 'light');
//    }
//}
//
//// Event listener for the dark mode toggle button
//document.getElementById('dark-mode-toggle').addEventListener('click', toggleDarkMode);
//
//// Check local storage for the saved theme preference on page load
//window.onload = function() {
//    if (localStorage.getItem('theme') === 'dark') {
//        document.body.classList.add('dark-mode');
//        document.querySelector('nav ul').classList.add('dark-mode');
//    }
//}

// document.addEventListener("DOMContentLoaded", function () {
//     const toggleButton = document.getElementById("dark-mode-toggle");
//     toggleButton.addEventListener("click", function () {
//         document.body.classList.toggle("dark-mode");
//     });
// });

document.addEventListener("DOMContentLoaded", function () {
    const toggleButton = document.getElementById("dark-mode-toggle");
    const icon = toggleButton.querySelector("i");

    toggleButton.addEventListener("click", function () {
        document.body.classList.toggle("dark-mode");
        if (document.body.classList.contains("dark-mode")) {
            icon.classList.remove("fa-moon");
            icon.classList.add("fa-sun");
        } else {
            icon.classList.remove("fa-sun");
            icon.classList.add("fa-moon");
        }
    });
});